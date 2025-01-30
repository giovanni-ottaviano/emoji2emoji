import os
import sys
import itertools
from typing import Dict
import torch
from torchinfo import summary

# Setting path
sys.path.append('..')

# -------------- Custom Module --------------- #
import utils.utils as ut
from utils.argparse_utils import create_parser
from data.data_loader import get_emoji_loader
from models import Generator, Discriminator
from loss import (
    discriminator_loss,
    generator_adversarial_loss,
    generator_cyclic_loss,
    generator_identity_loss
)


def training_loop(
    loader_X: torch.utils.data.DataLoader,
    loader_Y: torch.utils.data.DataLoader,
    test_loader_X: torch.utils.data.DataLoader,
    test_loader_Y: torch.utils.data.DataLoader,
    input_opts: Dict[str, str],
    use_cuda: bool
) -> None:

    """
        Run the training loop:
            - Update all the model parameters with standatd cycleGAN procedure
            - Save checkpoint with given frequency
            - Generate samples after every given iterations
    """

    # Create models
    D_X = Discriminator()
    D_Y = Discriminator()
    G_XY = Generator()
    G_YX = Generator()

    if input_opts['continue_training']:
        checkpoint = torch.load(input_opts['checkpoint_dir'] + '/optimizer.pkl')

        D_X.load_state_dict(torch.load(input_opts['checkpoint_dir'] + '/D_X.pkl'))
        D_Y.load_state_dict(torch.load(input_opts['checkpoint_dir'] + '/D_Y.pkl'))
        G_XY.load_state_dict(torch.load(input_opts['checkpoint_dir'] + '/G_XY.pkl'))
        G_YX.load_state_dict(torch.load(input_opts['checkpoint_dir'] + '/G_YX.pkl'))

    else:
        checkpoint = None

        D_X.apply(ut.weights_init_normal)
        D_Y.apply(ut.weights_init_normal)
        G_XY.apply(ut.weights_init_normal)
        G_YX.apply(ut.weights_init_normal)

    if input_opts['print_models_summary']:
        summary(G_XY, input_size=(input_opts['batch_size'], 3, input_opts['image_size'], input_opts['image_size']))
        summary(D_X,  input_size=(input_opts['batch_size'], 3, input_opts['image_size'], input_opts['image_size']))

    if use_cuda:
        G_XY.cuda()
        G_YX.cuda()
        D_X.cuda()
        D_Y.cuda()

        print('Models moved to GPU.')

    # Create optimizers
    g_params = itertools.chain(G_XY.parameters(), G_YX.parameters())
    d_params = itertools.chain(D_X.parameters(), D_Y.parameters())

    g_optimizer = torch.optim.Adam(g_params, input_opts['lr'], betas=[input_opts['beta1'], input_opts['beta2']])
    d_optimizer = torch.optim.Adam(d_params, input_opts['lr'], betas=[input_opts['beta1'], input_opts['beta2']])

    # Set epochs and lr scheduler for generator
    n_epochs = 1 if checkpoint is None else checkpoint['epoch']

    g_scheduler = ut.cycleGAN_scheduler(g_optimizer, n_epochs, input_opts['epoch_decay'])

    if input_opts['continue_training']:
        g_optimizer.load_state_dict(checkpoint['optimizer_state_dict_G'])
        d_optimizer.load_state_dict(checkpoint['optimizer_state_dict_D'])
        g_scheduler.load_state_dict(checkpoint['scheduler_state_dict_G'])


    # Data iters # GUARDA SE SI POSSONO CAMBIARE
    iter_X = iter(loader_X)
    iter_Y = iter(loader_Y)
    test_iter_X = iter(test_loader_X)
    test_iter_Y = iter(test_loader_Y)

    # Fixed data from both domains to inspect the model's performance
    device = 'cuda' if use_cuda else 'cpu'
    fixed_X = torch.tensor(next(test_iter_X)[0], device=device)
    fixed_Y = torch.tensor(next(test_iter_Y)[0], device=device)

    # Utils for training loop
    iters_per_epoch = min(len(iter_X), len(iter_Y))
    tot_iters = input_opts['epochs'] * iters_per_epoch
    iters_checkpoint = input_opts['checkpoint_every'] * iters_per_epoch
    iters_samples = input_opts['sample_every'] * iters_per_epoch

    for iteration in range(1, tot_iters + 1):
        if iteration % iters_per_epoch == 0:
            iter_X = iter(loader_X)
            iter_Y = iter(loader_Y)
            n_epochs += 1
            g_scheduler.step() # One step per epoch

        images_X, labels_X = next(iter_X)
        images_Y, labels_Y = next(iter_Y)
        images_X, labels_X = torch.tensor(images_X, device=device), torch.tensor(labels_X, device=device).long().squeeze()
        images_Y, labels_Y = torch.tensor(images_Y, device=device), torch.tensor(labels_Y, device=device).long().squeeze()

        D_X.train()
        D_Y.train()
        G_XY.train()
        G_YX.train()

        # Generator's training

        g_optimizer.zero_grad()

        # Compute the generator loss based on domain X
        fake_Y = G_XY(images_X)
        D_Y_out = D_Y(fake_Y)
        gXY_loss = generator_adversarial_loss(D_Y_out)  

        fake_X = G_YX(images_Y)
        D_X_out = D_X(fake_X)
        gYX_loss = generator_adversarial_loss(D_X_out) 

        cycle_X_loss, cycle_Y_loss = 0, 0
        if not input_opts['no_cycle_loss']:
            reconstructed_X = G_YX(fake_Y)
            reconstructed_Y = G_XY(fake_X)

            cycle_X_loss = generator_cyclic_loss(images_X, reconstructed_X, factor=input_opts['Lambda'])
            cycle_Y_loss = generator_cyclic_loss(images_Y, reconstructed_Y, factor=input_opts['Lambda'])


        identity_X_loss, identity_Y_loss = 0, 0
        if not input_opts['no_identity_loss']:
            identity_X = G_YX(images_X)
            identity_Y = G_XY(images_Y)

            identity_X_loss = generator_identity_loss(images_X, identity_X, factor=input_opts['Lambda'] * 0.5)
            identity_Y_loss = generator_identity_loss(images_Y, identity_Y, factor=input_opts['Lambda'] * 0.5)            


        g_loss = gXY_loss + gYX_loss + cycle_X_loss + cycle_Y_loss + identity_X_loss + identity_Y_loss
        g_loss.backward()
        g_optimizer.step()


        # Discriminator's training

        # Train with real images
        d_optimizer.zero_grad()

        # Compute loss idscriminator X
        fake_X = G_YX(images_Y)
        label_real_X = D_X(images_X)
        label_fake_X = D_X(fake_X.detach())

        D_X_loss = discriminator_loss(label_real_X, label_fake_X)

        fake_Y = G_XY(images_X)
        label_real_Y = D_Y(images_Y)
        label_fake_Y = D_Y(fake_Y.detach())

        D_Y_loss = discriminator_loss(label_real_Y, label_fake_Y)

        d_loss = D_X_loss + D_Y_loss
        d_loss.backward()
        d_optimizer.step()


        # Print the log info
        if iteration % input_opts['log_step'] == 0:
            print(f"Iteration [{iteration}/{tot_iters}] | D_X_loss: {D_X_loss.item():6.4f} | D_Y_loss: {D_Y_loss.item():6.4f} | G_loss: {g_loss.item():6.4f}")

        # Save the generated samples
        # if n_epochs % iters_samples == 0:
        if iteration % 100 == 0:
            ut.save_samples(iteration, G_XY, G_YX, fixed_X, fixed_Y, input_opts['batch_size'], input_opts['sample_dir'], use_cuda)

        # Save model's parameters
        if n_epochs % iters_checkpoint == 0:
            ut.make_checkpoint(n_epochs, G_XY, G_YX, D_X, D_Y, g_optimizer, d_optimizer, g_scheduler, input_opts['checkpoint_dir'])
            print(f"Checkpoint created at iteration {iteration} (epoch {n_epochs}).")



def main():

    """Run training on cycleGAN models for style transfer using the emoji dataset"""

    parser = create_parser()
    opts = parser.parse_args()

    ut.print_cl_options(opts)

    if opts.use_cuda and not torch.cuda.is_available():
        raise ValueError("Option 'use_cuda' was selected but no GPU found")

    # Initialize random number generators
    ut.initialize_RNGs(opts.seed, opts.use_cuda)

    # Create train and test dataloaders for images from the two domains X and Y
    loader_X, test_loader_X = get_emoji_loader(
        emoji_type=opts.start_style,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers
    )
    loader_Y, test_loader_Y = get_emoji_loader(
        emoji_type=opts.end_style,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers
    )

    # Create checkpoint and sample directories
    if not os.path.exists(opts.checkpoint_dir):
        os.makedirs(opts.checkpoint_dir)
        print(f"{opts.checkpoint_dir} directory created")
    else:
        print(f"{opts.checkpoint_dir} directory already exits. Skipping...")
    
    if not os.path.exists(opts.sample_dir):
        os.makedirs(opts.sample_dir)
        print(f"{opts.sample_dir} directory created")
    else:
        print(f"{opts.sample_dir} directory already exits. Skipping...")

    # Start training
    dict_input_opts = {
        'batch_size': opts.batch_size,
        'image_size': opts.image_size,
        'epochs': opts.epochs,
        'epoch_decay': opts.epoch_decay,
        'lr': opts.lr,
        'beta1': opts.beta1,
        'beta2': opts.beta2,
        'Lambda': opts.Lambda,
        'no_cycle_loss': opts.no_cycle_loss,
        'no_identity_loss': opts.no_identity_loss,
        'continue_training': opts.continue_training,
        'checkpoint_dir': opts.checkpoint_dir,
        'checkpoint_every': opts.checkpoint_every,
        'sample_every': opts.sample_every,
        'sample_dir': opts.sample_dir,
        'log_step': opts.log_step,
        'print_models_summary': opts.print_models_summary
    }

    training_loop(
        loader_X,
        loader_Y,
        test_loader_X,
        test_loader_Y,
        dict_input_opts,
        opts.use_cuda
    )



if __name__ == '__main__':
    main()