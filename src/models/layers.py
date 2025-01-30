# Implementation convolutional and deconvolutional layer
import torch
import torch.nn as nn


def convolutional_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int=2,
    padding: int=1,
    batch_norm: bool=True,
    init_zero_weights: bool=False
) -> nn.Sequential:
    
    """Return a convolutional layer with (optional) batch normalization"""

    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )

    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001

    layers = [conv_layer]

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


def deconvolutional_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int=2,
    padding: int=1,
    batch_norm: bool=True
) -> nn.Sequential:
    
    """Return a transposed-convolutional (deconvolutional) layer with (optional) batch normalization"""

    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)