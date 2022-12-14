# Inatall compressai -> pip install compressai

# Directory
    ./save -> checkpoint (.pth) file saved
    ./log  -> tensorboard log files saved
    ./data -> you can change

# Dataset Preparation
    Vimeo90k (82GB) for training (http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip)
    Kodak24
    For downloading Vimeo90K dataset, recommend using 'Axel' for parallel precessing

# train_mean_scale_hyperprior.py -> trainig
# test_mean_scale_hyperprior.py -> test
# custom_model.py -> You can customizing Mean-Scale Hyperprior from compressai project
# load_model.py -> customed load_model function from compressai project (for custom model)

1. train
    1) CUDA_VISIBLE_DEVICES={your_gpu_number} python train.py --quality {1-8}

    2) for each quality level, lambda is determined

2. test

    1) CUDA_VISIBLE_DEVICES={your_gpu_number} python test_mean_scale_hyperprior.py --quality {1-8}
               --checkpoint {your model dir}
    2) with no checkpoint, pre-trained model loaded

    3) validation is performed with first 1000 categories in Vimeo90K datasets

3. Arguments
    read som arguments in code for customizing

4. scheduling
    ReduceLROnPlateau is used and if learning rate is lower than 1e-6,training process stop early and save the last step.

5. best model is also saved but recommand to use last saved model weights

6. Model customizing

    1) forward() in CustomMeanScaleHyperprior class is used in training

    2) compress() and decompress() in CustomMeanScaleHyperprior class is used in inference (real entropy coding)

7. Configuration
    "-q", "--quality", type=int, default=0, help="quality of the model"
    "-save_dir", "--save_dir", type=str, default='save/', help="save_dir"
    "-log_dir", "--log_dir", type=str, default='log/', help="log_dir"
    "-total_step", default=5000000, type=int, help="total_step (default: %(default)s)"
    "-test_step", "--test_step", default=5000,
    "-save_step", default=100000,
    "-lr", "--learning-rate", default=1e-4,
    "-n", "--num-workers", type=int, default=4,
    "--patch-size", default=(256, 256),
    "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    "--test-batch-size", default=1, help="Test batch size (default: %(default)s)",
    "--aux-learning-rate", default=1e-3, help="Auxiliary loss learning rate (default: %(default)s)",
    "--checkpoint", type=str, help="Path to a checkpoint"

8.
    In log directory, enter this command: tensorboard dev upload --logdir ./

9.  There can be some errors.
    if any problem occurs, please contact me