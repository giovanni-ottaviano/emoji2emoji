import os
import argparse
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from tabulate import tabulate

# -------------- Custom Module --------------- #
from models import Generator, Discriminator


def weights_init_normal(model: torch.nn.Module) -> None:

    """Initialize Conv and BatchNorm2d layers with normally distributed weights"""

    name = model.__class__.__name__

    if name.find('Conv') != -1:
        torch.nn.init.normal_(model.weight.data, 0., 0.02)

    elif name.find('BatchNorm2d') != -1:
        torch.nn.init.normal(model.weight.data, 1., 0.02)
        torch.nn.init.constant(model.bias.data, 0.)


def initialize_RNGs(seed: int, use_cuda: bool) -> None:

    """Set seeds for random number generators (Numpy / PyTorch / CUDA)"""

    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda:
        torch.cuda.manual_seed(seed)


def cycleGAN_scheduler(optimizer, epoch_decay):

    """Return a scheduler with proper learning rate based on current training epoch """

    def lr_rule(epoch):

        # Needed for restarting the training
        weight = (epoch - epoch_decay) / (100 + 1.0)

        return 1. - max(0., weight)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_rule)

    return scheduler


def print_cl_options(opts: argparse.Namespace) -> None:

    """Print the given command-line arguments in tabular form"""
    
    list_opts = [(k, v) for k, v in vars(opts).items()]

    print(tabulate(list_opts, headers=["Parameter", "Value"], tablefmt="psql"))


def to_var(x, use_cuda: bool=False):

    """Converts numpy array to torch variable"""

    if use_cuda:
        x = x.cuda()

    return Variable(x)


def to_data(x, use_cuda: bool=False):

    """Converts torch ariable to numpy array"""

    if use_cuda:
        x = x.cpu()
        
    return x.data.numpy()


def make_checkpoint(
    epoch: int,
    G_XY: torch.nn.Module,
    G_YX: torch.nn.Module,
    D_X: torch.nn.Module,
    D_Y: torch.nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    scheduler_G: torch.optim.Optimizer,
    checkpoint_dir: Union[str, os.PathLike, bytes]
):
    
    """
        Saves the parameters of both generators (G_XY, G_YX) and discriminators (D_X, D_Y)
        on given checkpoint folder.
    """

    optim_dict = {
        'epoch'                  : epoch,
        'optimizer_state_dict_D' : optimizer_D.state_dict(),
        'optimizer_state_dict_G' : optimizer_G.state_dict(),
        'scheduler_state_dict_G' : scheduler_G.state_dict()
    }
     
    G_XY_path =  os.path.join(checkpoint_dir, 'G_XY.pkl')
    G_YX_path =  os.path.join(checkpoint_dir, 'G_YX.pkl')
    D_X_path =   os.path.join(checkpoint_dir, 'D_X.pkl')
    D_Y_path =   os.path.join(checkpoint_dir, 'D_Y.pkl')
    optim_path = os.path.join(checkpoint_dir, 'optimizer.pkl')

    torch.save(G_XY.state_dict(), G_XY_path)
    torch.save(G_YX.state_dict(), G_YX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)
    torch.save(optim_dict, optim_path)


def load_models(
    load: str,
    g_conv_dim: int,
    d_conv_dim: int,
    init_zero_weights: bool,
    use_cuda: bool=False
):
    
    """Loads the weights for generator and discriminator models from given checkpoints folder"""

    G_XY_path = os.path.join(load, 'G_XY.pkl')
    G_YX_path = os.path.join(load, 'G_YX.pkl')
    D_X_path =  os.path.join(load, 'D_X.pkl')
    D_Y_path =  os.path.join(load, 'D_Y.pkl')

    G_XY = Generator(conv_dim=g_conv_dim, init_zero_weights=init_zero_weights)
    G_YX = Generator(conv_dim=g_conv_dim, init_zero_weights=init_zero_weights)
    D_X = Discriminator(conv_dim=d_conv_dim)
    D_Y = Discriminator(conv_dim=d_conv_dim)

    G_XY.load_state_dict(torch.load(G_XY_path, map_location=lambda storage, loc: storage))
    G_YX.load_state_dict(torch.load(G_YX_path, map_location=lambda storage, loc: storage))
    D_X.load_state_dict(torch.load(D_X_path, map_location=lambda storage, loc: storage))
    D_Y.load_state_dict(torch.load(D_Y_path, map_location=lambda storage, loc: storage))

    if use_cuda:
        G_XY.cuda()
        G_YX.cuda()
        D_X.cuda()
        D_Y.cuda()

        print('Models moved to GPU.')


    return G_XY, G_YX, D_X, D_Y


# REWORK NEEDED FOR OPTIMIZATION
def merge_images(
    sources: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int
) -> torch.Tensor:

    """
        Return a grid consisting of pairs of images: the first is always the oringial one,
        whereas the second is the corresponding image generated by the CycleGAN.
    """
    _, _, h, w = sources.shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([3, row*h, row*w*2])

    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t

    return merged.transpose(1, 2, 0)/2. + 0.5


def save_samples(
    iteration: int,
    G_XY: torch.nn.Module,
    G_YX: torch.nn.Module,
    sample_X: torch.Tensor,
    sample_Y: torch.Tensor,
    batch_size: int,
    sample_dir: Union[str, bytes, os.PathLike],
    use_cuda: bool
):

    """Saves image samples from both generators"""

    fake_X = G_YX(sample_Y)
    fake_Y = G_XY(sample_X)

    X, fake_X = to_data(sample_X, use_cuda), to_data(fake_X, use_cuda)
    Y, fake_Y = to_data(sample_Y, use_cuda), to_data(fake_Y, use_cuda)

    merged_XY = merge_images(X, fake_Y, batch_size)
    merged_YX = merge_images(Y, fake_X, batch_size)

    path_XY = os.path.join(sample_dir, f'sample-{iteration}-X-Y.png')
    path_YX = os.path.join(sample_dir, f'sample-{iteration}-Y-X.png')

    plt.imsave(path_XY, merged_XY)
    plt.imsave(path_YX, merged_YX)

    print(f'Saved file {path_XY} and {path_YX}')



# # Helper function to show a batch - pytorch doc
# def show_landmarks_batch(sample_batched):
#     """Show image with landmarks for a batch of samples."""
#     images_batch, landmarks_batch = \
#             sample_batched['image'], sample_batched['landmarks']
#     batch_size = len(images_batch)
#     im_size = images_batch.size(2)
#     grid_border_size = 2

#     grid = utils.make_grid(images_batch)
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))

#     for i in range(batch_size):
#         plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
#                     landmarks_batch[i, :, 1].numpy() + grid_border_size,
#                     s=10, marker='.', c='r')

#         plt.title('Batch from dataloader')

# # if you are using Windows, uncomment the next line and indent the for loop.
# # you might need to go back and change ``num_workers`` to 0.

# # if __name__ == '__main__':
# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['landmarks'].size())

#     # observe 4th batch and stop.
#     if i_batch == 3:
#         plt.figure()
#         show_landmarks_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break