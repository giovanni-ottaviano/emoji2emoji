import argparse

# New type for argparse
def positiveint(value: str) -> int:

    """Type function for argparse - Positive int"""

    try:
        intvalue = int(value)
    except:
        raise argparse.ArgumentTypeError (f"{value} is not a valid positive int")

    if intvalue <= 0:
        raise argparse.ArgumentTypeError (f"{value} is not a valid positive int")

    return intvalue

# New type for argparse
def floatrange(lower_bound: float, upper_bound: float) -> float:

    """
        Type function for argparse - Float with bounds

         lower_bound - minimum acceptable parameter
         upper_bound - maximum acceptable parameter
    """

    # Define the function with default arguments
    def float_range_checker(value: str) -> float:

        try:
            f = float(value)
        except:
            raise argparse.ArgumentTypeError(f"{value} is not a valid float")

        if f < lower_bound or f > upper_bound:
            raise argparse.ArgumentTypeError(f"Argument must be in range [{lower_bound},{upper_bound}]")
        
        return f

    return float_range_checker

# AGGIUNGI LOAD E SAMPLE EVERY
def create_parser() -> argparse.Namespace:

    """Return a parser for command line inputs"""

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=positiveint, default=128, help='Side lenth of the converted image (N x N)')
    parser.add_argument('--g_conv_dim', type=positiveint, default=64)
    parser.add_argument('--d_conv_dim', type=positiveint, default=32)
    parser.add_argument('--no_cycle_loss', action='store_true', default=False, help='Choose whether not to include the cycle consistency term in the loss function')
    parser.add_argument('--no_identity_loss', action='store_true', default=False, help='Choose whether not to include the identity term in the loss function')
    parser.add_argument('--init_zero_weights', action='store_true', default=False, help='Choose whether to initialize the generator weights to 0')

    # Training hyper-parameters
    parser.add_argument('--continue_training', action='store_true', default=False, help='Whether or not to continue the training (loading weights from checkpoint directory)')    
    parser.add_argument('--epochs', type=positiveint, default=200, help='Number of training epochs.')
    parser.add_argument('--epoch_decay', type=positiveint, default=100, help='The epoch where weight decay of learning rate starts')
    parser.add_argument('--batch_size', type=positiveint, default=16, help='Number of images per batch')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of threads to use in the DataLoader')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta 1 parameter for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta 2 parameter for Adam optimizer')
    parser.add_argument('--Lambda', type=float, default=10., help='The (positive) weight for cycle loss')
    parser.add_argument('--seed', type=positiveint, default=123, help='Seed for random number generetors')
    parser.add_argument('--use_cuda', action='store_true', default=False, help='Whether or not to run training on GPU')

    # Data sources
    parser.add_argument('--start_style', type=str, default='Apple', choices=['Apple', 'Windows'], help='Choose the type of images for starting domain')
    parser.add_argument('--end_style', type=str, default='Windows', choices=['Apple', 'Windows'], help='Choose the type of images for ending domain')

    # Checkpoint and printing options
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_cyclegan', help='Path to the checkpoint directory')
    parser.add_argument('--sample_dir', type=str, default='samples_cyclegan', help='Path to the samples directory')
    parser.add_argument('--load', type=str, default=None, help='') # MANCA
    parser.add_argument('--log_step', type=positiveint, default=10, help='Number of steps before printing a log')
    parser.add_argument('--sample_every', type=positiveint, default=1, help='') # MANCA
    parser.add_argument('--checkpoint_every', type=positiveint, default=1, help='Number of steps before checkpoint')
    parser.add_argument('--print_models_summary', action='store_true', default=False, help='Whether or not to print a short summary of model layers and parameters')    

    return parser