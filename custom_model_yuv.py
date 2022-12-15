from compressai.models import MeanScaleHyperprior
from torch import nn, torch
from compressai.models.utils import conv, deconv
from torchvision.transforms import transforms, functional


class CustomMeanScaleHyperprior(MeanScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.t_a_zero_luma = nn.Sequential(
            conv(1, N),
            nn.PReLU(),
        )

        self.t_a_zero_luma_inv = nn.Sequential(
            nn.PReLU(),
            deconv(N, 1),
        )

        self.t_a_zero_chroma = nn.Sequential(
            conv(2, N, kernel_size=3, stride=1),
            nn.PReLU(),
        )

        self.t_a_zero_chroma_inv = nn.Sequential(
            nn.PReLU(),
            conv(N, 2, kernel_size=3, stride=1),
        )

        self.t_a_1by1 = nn.Sequential(
            conv(2*N, N, stride=1, kernel_size=1),
            nn.PReLU(),
        )
        
        self.t_a_1by1_inv = nn.Sequential(
            nn.PReLU(),
            conv(N, 2*N, stride=1, kernel_size=1),
        )

        self.t_a_r = nn.Sequential(
            conv(N, N),
            nn.PReLU(),
            conv(N, N),
            nn.PReLU(),
            conv(N, M),
        )

        self.t_a_r_inv = nn.Sequential(
            deconv(M, N),
            nn.PReLU(),
            deconv(N, N),
            nn.PReLU(),
            deconv(N, N),
        )

    def t_a(self, luma, chroma):
        t_zero_luma = self.t_a_zero_luma(luma)
        down_sampling = transforms.Compose([
            transforms.Resize((chroma.shape[2] // 2, chroma.shape[3] // 2)),
        ])
        chroma = down_sampling(chroma)
        t_zero_chroma = self.t_a_zero_chroma(chroma)
        t_1by1 = self.t_a_1by1(torch.cat([t_zero_luma, t_zero_chroma], dim=1))
        y = self.t_a_r(t_1by1)
        return y

    def t_a_inv(self, y_hat):
        t_r_inv = self.t_a_r_inv(y_hat)
        t_1by1_inv = self.t_a_1by1_inv(t_r_inv)
        luma, chroma = torch.chunk(t_1by1_inv, chunks = 2, dim = 1)
        t_zero_luma_inv = self.t_a_zero_luma_inv(luma)
        t_zero_chroma_inv = self.t_a_zero_chroma_inv(chroma)
        up_sampling = transforms.Compose([
            transforms.Resize((t_zero_chroma_inv.shape[2] * 2, t_zero_chroma_inv.shape[3] * 2)),
        ])
        t_zero_chroma_inv = up_sampling(t_zero_chroma_inv)
        x_hat = torch.cat([t_zero_luma_inv, t_zero_chroma_inv], dim = 1)
        return x_hat

    def forward(self, luma, chroma):
        y = self.t_a(luma, chroma)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.t_a_inv(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, luma, chroma):
        y = self.t_a(luma, chroma)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.t_a_inv(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
