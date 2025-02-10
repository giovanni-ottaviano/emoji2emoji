# emoji2emoji - Style transfer with deep cycleGANs :pushpin:

## Overview
This repository contains the code to train and test a deep *CycleGAN* for style transfer on images of emojis.\
The training process is highly customizable, allowing full control on the losses’ and optimizers’ parameters
and making possible to split the training process in different runs. The learning rate is fixed for a customizable number of epochs
and subsequently it decreases with a Lambda scheduling.

This framework offers a wide variety of styles, such as Apple, Windows and Facebook, and it is also possible to add new ones
with a simple modification of the input parameters, as well as adding the relevant dataset.


## Usage
The folder ```src``` contains the main source code for this project, separated by specific folders.\
the training routine is included in ```train_model.py```, in the ```src/model``` folder, and requires a bunch of non-standard
python modules; it is recommended to create a new virtual environment, for example using *conda* or *venv*, and install all the required
packages report in the file ```requirements.txt```.\
Here, there's a comprehensive list of required and optional input parameters for the training routine:
```
optional arguments:
  -h, --help            show this help message and exit
  --image_size
                        Side lenth of the converted image (N x N)
  --g_conv_dim
  --d_conv_dim
  --no_cycle_loss       Choose whether not to include the cycle consistency term in the loss
                        function
  --no_identity_loss    Choose whether not to include the identity term in the loss function
  --init_zero_weights   Choose whether to initialize the generator weights to 0
  --continue_training   Whether or not to continue the training (loading weights from checkpoint directory)
  --epochs              Number of training epochs.
  --epoch_decay
                        The epoch where weight decay of learning rate starts
  --batch_size
                        Number of images per batch
  --num_workers
                        Number of threads to use in the DataLoader
  --lr                  Learning rate
  --beta1               Beta 1 parameter for Adam optimizer
  --beta2               Beta 2 parameter for Adam optimizer
  --Lambda              The (positive) weight for cycle loss
  --seed                for random number generetors
  --use_cuda            Whether or not to run training on GPU
  --start_style
                        Choose the type of images for starting domain
  --end_style
                        Choose the type of images for ending domain
  --checkpoint_dir
                        Path to the checkpoint directory
  --sample_dir
                        Path to the samples directory
  --load_dir            Path to the checkpoint directory (used only if continue_training parameter is given)
  --log_step            Number of steps before printing a log
  --checkpoint_every
                        Number of epochs before checkpoint and sample generation
  --print_models_summary
                        Whether or not to print a short summary of model layers and parameters
```

## Examples
The folder ```samples``` contains a set of training samples in different epochs of the trainig process for different choices of styles.

Instead the folder ```models``` comprehends all the trained weights for several choices of emoji style

## Bibliography
[1] Zhu et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks". (2017) URL: https://doi.org/10.48550/arXiv.1703.10593.
Repository ([link](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix))


## Acknowledgement
The idea and basic structure of the code were taken from an assignement of the *Neural Networks and Machine Learning* course from the University of Toronto.