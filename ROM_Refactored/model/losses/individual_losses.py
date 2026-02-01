"""
Individual loss calculation functions
These are used by CustomizedLoss to compute different loss components
"""

import torch
import torch.nn as nn


def get_reconstruction_loss(x, t_decoded, reconstruction_variance=0.1):
    """
    Calculate reconstruction loss with configurable variance parameter.
    
    Args:
        x: True spatial state tensor
        t_decoded: Reconstructed spatial state tensor  
        reconstruction_variance: Assumed variance of reconstruction noise (from config)
                               Lower values = stricter reconstruction demands
                               Higher values = more forgiving reconstruction tolerance
    
    Returns:
        Reconstruction loss normalized by the expected noise level
    """
    v = reconstruction_variance
    return torch.mean(torch.sum((x.reshape(x.size(0), -1) - t_decoded.reshape(t_decoded.size(0), -1)) ** 2 / (2*v), dim=-1))


def get_l2_reg_loss(qm):
    """Calculate L2 regularization loss"""
    l2_reg = 0.5 * qm.pow(2)
    return torch.mean(torch.sum(l2_reg, dim=-1))


def get_flux_loss(state, state_pred, channel_mapping):
    """
    Calculate flux conservation loss using configurable channel indices
    
    Args:
        state: True state tensor [batch, channels, X, Y, Z]
        state_pred: Predicted state tensor [batch, channels, X, Y, Z]
        channel_mapping: Dictionary with channel indices for each physical field
        
    Returns:
        flux_loss: Mean absolute error of flux conservation in X, Y, Z directions
    """
    # Extract channel indices from configuration
    pressure_ch = channel_mapping['pressure']
    use_precomputed = channel_mapping.get('use_precomputed_trans', False)
    
    # Extract pressure fields
    p = state[:, pressure_ch, :, :, :].unsqueeze(1)          # [batch, 1, X, Y, Z]
    p_pred = state_pred[:, pressure_ch, :, :, :].unsqueeze(1)  # [batch, 1, X, Y, Z]
    
    total_flux_loss = 0.0
    num_directions = 0
    
    # ===== X-DIRECTION FLUX =====
    if p.size(2) > 1:  # Check if X dimension > 1
        if use_precomputed and 'trans_x' in channel_mapping:
            # Use precomputed transmissibilities
            trans_x_ch = channel_mapping['trans_x']
            tran_x = state[:, trans_x_ch, 1:, :, :].unsqueeze(1)  # [batch, 1, X-1, Y, Z]
        else:
            # Calculate transmissibilities from permeabilities
            perm_x_ch = channel_mapping['perm_x']
            perm_x = state[:, perm_x_ch, :, :, :].unsqueeze(1)
            # Harmonic average between adjacent cells
            tran_x = 2.0 / (1.0 / perm_x[:, :, 1:, :, :] + 1.0 / perm_x[:, :, :-1, :, :])
        
        # Calculate fluxes using Darcy's law
        flux_x = (p[:, :, 1:, :, :] - p[:, :, :-1, :, :]) * tran_x
        flux_x_pred = (p_pred[:, :, 1:, :, :] - p_pred[:, :, :-1, :, :]) * tran_x
        
        # Compute L1 loss
        loss_x = torch.mean(torch.abs(flux_x - flux_x_pred))
        total_flux_loss += loss_x
        num_directions += 1
    
    # ===== Y-DIRECTION FLUX =====
    if p.size(3) > 1:  # Check if Y dimension > 1
        if use_precomputed and 'trans_y' in channel_mapping:
            # Use precomputed transmissibilities
            trans_y_ch = channel_mapping['trans_y']
            tran_y = state[:, trans_y_ch, :, 1:, :].unsqueeze(1)  # [batch, 1, X, Y-1, Z]
        else:
            # Calculate transmissibilities from permeabilities
            perm_y_ch = channel_mapping['perm_y']
            perm_y = state[:, perm_y_ch, :, :, :].unsqueeze(1)
            # Harmonic average between adjacent cells
            tran_y = 2.0 / (1.0 / perm_y[:, :, :, 1:, :] + 1.0 / perm_y[:, :, :, :-1, :])
        
        # Calculate fluxes using Darcy's law
        flux_y = (p[:, :, :, 1:, :] - p[:, :, :, :-1, :]) * tran_y
        flux_y_pred = (p_pred[:, :, :, 1:, :] - p_pred[:, :, :, :-1, :]) * tran_y
        
        # Compute L1 loss
        loss_y = torch.mean(torch.abs(flux_y - flux_y_pred))
        total_flux_loss += loss_y
        num_directions += 1
    
    # ===== Z-DIRECTION FLUX =====
    if p.size(4) > 1:  # Check if Z dimension > 1
        if use_precomputed and 'trans_z' in channel_mapping:
            # Use precomputed transmissibilities
            trans_z_ch = channel_mapping['trans_z']
            tran_z = state[:, trans_z_ch, :, :, 1:].unsqueeze(1)  # [batch, 1, X, Y, Z-1]
        else:
            # Calculate transmissibilities from permeabilities
            perm_z_ch = channel_mapping['perm_z']
            perm_z = state[:, perm_z_ch, :, :, :].unsqueeze(1)
            # Harmonic average between adjacent cells
            tran_z = 2.0 / (1.0 / perm_z[:, :, :, :, 1:] + 1.0 / perm_z[:, :, :, :, :-1])
        
        # Calculate fluxes using Darcy's law  
        flux_z = (p[:, :, :, :, 1:] - p[:, :, :, :, :-1]) * tran_z
        flux_z_pred = (p_pred[:, :, :, :, 1:] - p_pred[:, :, :, :, :-1]) * tran_z
        
        # Compute L1 loss
        loss_z = torch.mean(torch.abs(flux_z - flux_z_pred))
        total_flux_loss += loss_z
        num_directions += 1
    
    # Average loss across all computed directions
    if num_directions > 0:
        flux_loss = total_flux_loss / num_directions
    else:
        flux_loss = torch.tensor(0.0, device=state.device)
    
    return flux_loss


def get_binary_sat_loss(state, state_pred):
    """Calculate binary saturation loss"""
    sat_threshold = 0.105
    sat = state[:, :, :, 0].unsqueeze(-1)
    sat_pred = state_pred[:, :, :, 0].unsqueeze(-1)

    sat_bool = sat >= sat_threshold
    sat_bin = sat_bool.float()

    sat_pred_bool = sat_pred >= sat_threshold
    sat_pred_bin = sat_pred_bool.float()

    binary_loss = nn.functional.binary_cross_entropy(sat_pred_bin, sat_bin)
    return torch.mean(binary_loss)


def get_non_negative_loss(reconstructed_states, predicted_observations):
    """
    Calculate non-negativity constraint loss for physical realism
    
    Args:
        reconstructed_states: List of reconstructed spatial states
        predicted_observations: List of predicted observation values
        
    Returns:
        non_negative_loss: Mean squared penalty for negative values
    """
    total_loss = 0.0
    num_terms = 0
    
    # Penalize negative values in reconstructed spatial states
    for state in reconstructed_states:
        negative_values = torch.clamp(-state, min=0)  # Only negative parts
        if negative_values.numel() > 0:
            total_loss += torch.mean(negative_values ** 2)
            num_terms += 1
    
    # Penalize negative values in predicted observations
    for obs in predicted_observations:
        negative_values = torch.clamp(-obs, min=0)  # Only negative parts
        if negative_values.numel() > 0:
            total_loss += torch.mean(negative_values ** 2)
            num_terms += 1
    
    return total_loss / max(num_terms, 1)  # Average over all terms


def get_well_bhp_loss(state, state_pred, prod_well_loc, pressure_channel=2):
    """
    Calculate BHP loss for wells at specified locations
    
    Args:
        state: True state tensor [batch, channels, X, Y, Z]
        state_pred: Predicted state tensor [batch, channels, X, Y, Z]  
        prod_well_loc: Well locations tensor [num_wells, 2] with [X, Y] coordinates
        pressure_channel: Index of pressure channel in state tensor (default: 2)
        
    Returns:
        bhp_loss: Mean absolute error of pressure predictions at well locations
    """
    # Extract pressure channel at well locations
    # For 3D tensors, we take the average pressure across all Z layers for each well
    batch_size = state.shape[0]
    num_wells = prod_well_loc.shape[0]
    
    p_true_wells = []
    p_pred_wells = []
    
    for i in range(num_wells):
        x_coord = prod_well_loc[i, 0]
        y_coord = prod_well_loc[i, 1]
        
        # Extract pressure at well location across all Z layers (penetrating all layers)
        # Take mean across Z dimension to get average BHP
        p_true_well = torch.mean(state[:, pressure_channel, x_coord, y_coord, :], dim=-1)  # [batch]
        p_pred_well = torch.mean(state_pred[:, pressure_channel, x_coord, y_coord, :], dim=-1)  # [batch]
        
        p_true_wells.append(p_true_well)
        p_pred_wells.append(p_pred_well)
    
    # Stack all wells: [batch, num_wells]
    p_true = torch.stack(p_true_wells, dim=1)
    p_pred = torch.stack(p_pred_wells, dim=1)
    
    # Calculate mean absolute error across all wells and batches
    bhp_loss = torch.mean(torch.abs(p_true - p_pred))
    return bhp_loss


def get_kl_divergence_loss(mu, logvar, reduction='mean'):
    """
    KL divergence loss for Variational Autoencoder.
    Computes KL(q(z|x) || p(z)) where q(z|x) = N(mu, sigma^2) and p(z) = N(0, I).
    
    KL divergence formula:
    KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    Args:
        mu: Mean of the latent distribution (batch_size, latent_dim)
        logvar: Log variance of the latent distribution (batch_size, latent_dim)
        reduction: 'sum', 'mean', or 'none'
                  - 'sum': Sum over all dimensions and batch
                  - 'mean': Mean over batch (sum over latent dims, mean over batch)
                  - 'none': Return per-sample KL (batch_size,)
    
    Returns:
        KL divergence loss
    """
    # KL divergence per sample: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    
    if reduction == 'sum':
        return kl_per_sample.sum()
    elif reduction == 'mean':
        return kl_per_sample.mean()
    elif reduction == 'none':
        return kl_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Use 'sum', 'mean', or 'none'.")

