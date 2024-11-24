import torch
import torch.nn as nn


def discriminator_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor, factor: float=0.5):

    """Returns the loss for the discriminator (on both real and fake data)"""

    criterion = nn.MSELoss() # or nn.BCEWithLogitsLoss() depending on the discriminator
    target_real = torch.ones_like(pred_real)
    target_fake = torch.zeros_like(pred_fake)

    loss_real = criterion(target_real, pred_real)
    loss_fake = criterion(target_fake, pred_fake)

    return (loss_real + loss_fake) * factor


def generator_adversarial_loss(pred_fake: torch.Tensor, factor: float=1.):

    """Returns the adversarial part of the generator loss"""

    criterion = nn.MSELoss()
    target = torch.ones_like(pred_fake)

    return criterion(pred_fake, target) * factor


def generator_cyclic_loss(real: torch.Tensor, cyclic: torch.Tensor, factor: float=10.):

    """Returns the cyclic part of the generator loss (Cycle Consistency Loss)"""

    criterion = nn.L1Loss()
    loss = criterion(real, cyclic)

    return loss * factor


def generator_identity_loss(real: torch.Tensor, identity: torch.Tensor, factor: float=5.):

    """Returns the identity part of the generator loss"""

    criterion = nn.L1Loss()
    loss = criterion(real, identity)

    return loss * factor