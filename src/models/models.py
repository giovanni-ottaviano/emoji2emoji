
# https://arxiv.org/pdf/1703.10593.pdf
# https://arxiv.org/pdf/1611.07004.pdf

# Implementation of ResidualBlock, Generator and Discriminator (PatchGAN)
import torch.nn as nn


class ResidualNetBlock(nn.Module):

    """Define a ResNet block (i.e. a convolutional block with skip connections)"""

    def __init__(self, in_channels):
        super(ResidualNetBlock, self).__init__()

        self.convolutional_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )


    def forward(self, x):

        """Forward pass (with skip connection)"""

        return x + self.convolutional_block(x)



class Discriminator(nn.Module):

    """Discriminator for generative architecture (PatchGAN)"""

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        filters = 64
        kernel_size = 4
        stride = 2
        padding = 1
        negative_slope = 0.2

        model = [
            nn.Conv2d(in_channels, filters, kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(negative_slope, inplace=True),
        ]

        for _ in range(1,4):
            filters *= 2 

            model += [
                nn.Conv2d(filters // 2, filters, kernel_size, stride=stride, padding=padding),
                nn.InstanceNorm2d(128), 
                nn.LeakyReLU(negative_slope, inplace=True),
            ]

        model += [nn.Conv2d(filters, 1, kernel_size, padding=padding)]
        self.model = nn.Sequential(*model)
    

    def forward(self, x):

        """Forward pass"""

        return self.model(x)


class Generator(nn.Module):

    """Generator class for cyclic generative adversarial architecture (CycleGAN)"""

    def __init__(self, in_channels=3, ncf=64):
        super(Generator, self).__init__()

        n_resblocks = 6

        # Initial convolutional block       
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ncf, 7),
            nn.InstanceNorm2d(ncf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = ncf
        out_features = ncf * 2

        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_resblocks):
            model += [ResidualNetBlock(in_features)]

        # Upsampling
        out_features = in_features // 2

        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, 3, 7),
            nn.Tanh()
        ]

        # Set sequential model
        self.model = nn.Sequential(*model)


    def forward(self, x):

        """Forward pass"""

        return self.model(x)