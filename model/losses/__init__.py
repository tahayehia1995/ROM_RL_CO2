# Loss functions module

from .spatial_enhancements import GradientLoss, Discriminator3D
from .individual_losses import (
    get_reconstruction_loss,
    get_flux_loss,
    get_well_bhp_loss,
    get_l2_reg_loss,
    get_non_negative_loss,
    get_binary_sat_loss
)
from .customized_loss import CustomizedLoss

__all__ = [
    'GradientLoss',
    'Discriminator3D',
    'get_reconstruction_loss',
    'get_flux_loss',
    'get_well_bhp_loss',
    'get_l2_reg_loss',
    'get_non_negative_loss',
    'get_binary_sat_loss',
    'CustomizedLoss'
]
