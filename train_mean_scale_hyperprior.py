# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import math
import random
import shutil
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from load_model import load_model
from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.zoo import image_models
from dataset import ImageFolderVimeo, Kodak24Dataset
from tensorboardX import SummaryWriter


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    '''
    "bmshj2018-factorized": bmshj2018_factorized,
    "bmshj2018-hyperprior": bmshj2018_hyperprior,
    "mbt2018-mean": mbt2018_mean,
    "mbt2018": mbt2018,
    "cheng2020-anchor": cheng2020_anchor,
    "cheng2020-attn": cheng2020_attn,
    '''
    parser.add_argument(
        "-m",
        "--model",
        default="mbt2018-mean",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-q", "--quality", type=int, default=0, help="quality of the model"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default='../data', help="Training dataset"
    )
    parser.add_argument(
        "-save_dir", "--save_dir", type=str, default='save/', help="save_dir"
    )
    parser.add_argument(
        "-log_dir", "--log_dir", type=str, default='log/', help="log_dir"
    )
    parser.add_argument(
        "-total_step",
        "--total_step",
        default=5000000,
        type=int,
        help="total_step (default: %(default)s)",
    )
    parser.add_argument(
        "-test_step",
        "--test_step",
        default=5000,
        type=int,
        help="test_step (default: %(default)s)",
    )
    parser.add_argument(
        "-save_step",
        "--save_step",
        default=100000,
        type=int,
        help="save_step (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    # parser.add_argument(
    #     "--lambda",
    #     dest="lmbda",
    #     type=float,
    #     default=0.0018,
    #     # quality   :     1        2        3        4        5        6       7       8
    #     # lambda    :  0.0018   0.0035   0.0067   0.0130   0.0250   0.0483  0.0932   0.1800
    #     help="Bit-rate distortion parameter (default: %(default)s)",
    # )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=123, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, quality):
        super().__init__()
        qualities_to_lambda = {
            1: 0.0018,
            2: 0.0035,
            3: 0.0067,
            4: 0.0130,
            5: 0.0250,
            6: 0.0483,
            7: 0.0932,
            8: 0.1800
        }
        self.mse = nn.MSELoss()
        self.lmbda = qualities_to_lambda[quality]

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = { #hyperprior??? 
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def test(global_step, test_dataloader, model, criterion, logger, lr):
    model.eval()
    model.update(force=True)
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for i_, d in enumerate(test_dataloader):
            d = d.to(device)

            luma = d[0,0,:,:].unsqueeze(dim=0).unsqueeze(dim=0)
            chroma = d[0,1:,:,:].unsqueeze(dim=0)

            out_net = model(luma, chroma)
            # out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test global_step {global_step}: Average losses:"
        f"\tTest Loss: {loss.avg:.3f} |"
        f"\tTest MSE loss: {mse_loss.avg:.3f} |"
        f"\tTest Bpp loss: {bpp_loss.avg:.2f} |"
        f"\tTest Aux loss: {aux_loss.avg:.2f}\n"
    )

    logger.add_scalar('Test Loss', loss.avg, global_step)
    logger.add_scalar('Test MSE loss', mse_loss.avg, global_step)
    logger.add_scalar('Test Bpp loss', bpp_loss.avg, global_step)
    logger.add_scalar('Test Aux loss', aux_loss.avg, global_step)
    logger.add_scalar('lr', lr, global_step)

    return loss.avg


def train(
        model, criterion, train_dataloader, test_dataloader, optimizer, aux_optimizer, lr_scheduler, global_step,
        args, logger):
    model.train()
    device = next(model.parameters()).device
    best_loss = float("inf")

    for loop in range(1000):  # infinite loop
        for i, x in enumerate(tqdm(train_dataloader)):
            global_step += 1
            x = x.to(device)

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            luma = x[:,0,:,:].unsqueeze(dim=1)
            chroma = x[:,1:,:,:]

            out_net = model(luma, chroma)
            # out_net = model(x)

            out_criterion = criterion(out_net, x)

            out_criterion["loss"].backward()

            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            # Training log
            if global_step % 100 == 0:
                tqdm.write(
                    f"Train step \t{global_step}: \t["
                    f"{i * len(x)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )

                logger.add_scalar('Loss', out_criterion["loss"].item(), global_step)
                logger.add_scalar('MSE loss', out_criterion["mse_loss"].item(), global_step)
                logger.add_scalar('Bpp loss', out_criterion["bpp_loss"].item(), global_step)
                logger.add_scalar('Aux loss', aux_loss.item(), global_step)

            # validation
            if global_step % args.test_step == 0:
                loss = test(global_step, test_dataloader, model, criterion, logger, optimizer.param_groups[0]['lr'])
                model.train()

                lr_scheduler.step(loss)

                is_best = loss < best_loss
                if is_best:
                    print("!!!!!!!!!!!BEST!!!!!!!!!!!!!")
                best_loss = min(loss, best_loss)

                if global_step % args.save_step == 0:
                    save_checkpoint(
                        {
                            "global_step": global_step,
                            "state_dict": model.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        },
                        is_best, #best????????? ?????? ???????????? ???
                        filename=f'{args.save_dir}/{global_step}_checkpoint.pth'
                    )

                # Early stop // learning rate??? 1e-6?????? ????????????
                if optimizer.param_groups[0]['lr'] < 1e-6:
                    print(f'Finished. \tcurrent lr:{optimizer.param_groups[0]["lr"]} \tglobal step:{global_step}')
                    save_checkpoint(
                        {
                            "global_step": global_step,
                            "state_dict": model.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        },
                        is_best,
                        filename=f'{args.save_dir}/{global_step}_checkpoint.pth'
                    )
                    exit(0)


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.split('/')[0] + "/checkpoint_best_loss.pth")


def build_dataset(args):
    train_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomCrop(args.patch_size)]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = ImageFolderVimeo(args.dataset, transform=train_transforms)
    test_dataset = Kodak24Dataset(args.dataset, transform=test_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    print('Dataset load')
    train_dataloader, test_dataloader = build_dataset(args)
    logger = SummaryWriter(args.log_dir)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print('Device : ', device)
    print('args : ', args)
    print('Model load')
    #net = load_model(args.model, metric="mse", quality=args.quality, pretrained=True)
    net = load_model(args.model, metric="mse", quality=args.quality, pretrained=False)
    net = net.to(device)
    global_step = 0

    if args.cuda and torch.cuda.device_count() > 1:
        print('Using Multiple GPU')
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                        patience=round(1e5 / args.test_step) - 1)
    criterion = RateDistortionLoss(quality=args.quality)

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        global_step = checkpoint["global_step"]
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

    train(net,
        criterion,
        train_dataloader,
        test_dataloader,
        optimizer,
        aux_optimizer,
        lr_scheduler,
        global_step,
        args,
        logger)


if __name__ == "__main__":
    main(sys.argv[1:])
