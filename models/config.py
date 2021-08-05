"""Configuration (arguments) update functions."""
# Import modules
from torch.cuda import is_available as gpu_available
from torch import manual_seed as torch_manual_seed
from torch import set_deterministic
from torch.backends import cudnn as cudnn
from numpy.random import seed as npseed
from random import seed as randomseed
from argparse import Namespace
from numpy import prod


def update_common_args(args: Namespace) -> Namespace:
    # Declare the common arguments
    # The mode
    args.mode = 'main'
    # The model
    args.model = 'plain'
    # The batch size
    args.bs = 1
    # Whether to use all data as labeled
    args.full_data_train = True
    # The number of epochs
    if args.test_mode:
        args.epochs = 2
    elif args.data == 'Colon':
        args.epochs = 100
    elif args.data == 'MNIST':
        args.epochs = 100
    # The seed
    args.seed = 71
    # The beta [KL div]
    args.beta = 1
    # The number of samples
    args.numsamp = 10
    # Enable/Disable CUDA
    args.device = 'cuda' if gpu_available() else 'cpu'
    # Enable cuDNN auto-tuner if it is MNIST set
    cudnn.benchmark = True if args.data == 'MNIST' else False
    # # The reproducibility
    # args.reproducibility = True
    # # Use deterministic algorithms for convolutional operations
    # set_deterministic(args.reproducibility)
    # Set seed in torch, numpy and built-in random modules
    torch_manual_seed(args.seed)
    npseed(args.seed)
    randomseed(args.seed)
    # Return the arguments
    return args


def update_args_MNIST(args: Namespace) -> Namespace:
    # Declare data dimensions
    # The reconstruction dimension
    args.recon_dims = (1, 28, 28)
    # The flattened input dimension
    args.x_dim = prod(args.recon_dims)
    # The number of input channels
    args.input_channels = args.recon_dims[0]
    # The number of instances in a bag
    args.mean_bag_length = 10
    # The number of labeled data
    args.n_labeled = 10000
    # The target number of the bag
    args.target_number = 9
    # Declare the model dimensions
    # The hidden dimensions
    args.hidden_dims = [32, 64, 128, 256, 512]
    # Declare the image dimensions in the latent space
    args.latent_img_size = (1, 1)
    # The number of encoder output channels
    args.enc_out = 16 if args.model != 'disent' else 48
    # The number of decoder input channels
    args.dec_inp = 16 if args.model != 'disent' else 32
    # Declare the alpha multiplier
    args.alpha_multiplier = 5.
    # Declare alpha value
    args.alpha = args.alpha_multiplier * args.dec_inp * \
        0.1 * (60000 - args.n_labeled) / args.n_labeled
    # The learning rate
    args.lr = 1e-4
    # The weight decay
    args.wd = 1e-3
    # Return the arguments
    return args


def update_args_Colon(args: Namespace) -> Namespace:
    # Declare data dimensions
    # Whether to apply padding
    args.padding = False if args.model == 'base_att' else True
    # The reconstruction dimension
    args.recon_dims = (3, 28, 28) if args.padding else (3, 27, 27)
    # The flattened input dimension
    args.x_dim = prod(args.recon_dims)
    # The number of input channels
    args.input_channels = args.recon_dims[0]
    # The number of labeled data
    args.n_labeled = 22
    # Declare the number of components in the loss calculation
    args.num_components = 5
    # Declare the model dimensions
    # The hidden dimensions
    args.hidden_dims = [64, 128]
    # The kernel sizes
    args.kernel_sizes = [5, 4, 3]
    # Declare the image dimensions in the latent space
    args.latent_img_size = (2, 2)
    # The number of encoder output channels
    args.enc_out = 32 if args.model != 'disent' else 48
    # The number of input channels of the decoder
    args.dec_inp = int(args.enc_out / 2)
    # The number of output channels of the decoder
    args.dec_out = 100
    # Declare alpha value
    args.alpha = 500
    # The learning rate
    args.lr = 1e-4
    # The weight decay
    args.wd = 3e-5
    # Return the arguments
    return args


def update_args(args: Namespace) -> Namespace:
    # Update the common arguments
    args = update_common_args(args=args)
    # Update the arguments according to the dataset
    args = update_args_MNIST(args=args) if args.data == 'MNIST' \
        else update_args_Colon(args=args)
    # Return the arguments
    return args
