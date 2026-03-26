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


def get_fft_loss(x, x_pred, normalize=True):
    """
    Calculate FFT loss - L2 distance in frequency domain.
    Forces model to preserve both low and high frequency components.
    
    This loss transforms both the input and reconstruction into the frequency
    domain using the Fast Fourier Transform (FFT). The L2 distance between
    the two transformed sets is then calculated, forcing the model to directly
    minimize errors between high- and low-frequency components.
    
    Args:
        x: True spatial state tensor [batch, channels, X, Y, Z]
        x_pred: Predicted/reconstructed spatial state tensor [batch, channels, X, Y, Z]
        normalize: Whether to normalize by number of frequency components (default: True)
    
    Returns:
        FFT loss (L2 norm of FFT difference)
    """
    # Apply 3D FFT to both tensors across spatial dimensions
    # dim=(-3, -2, -1) applies FFT to X, Y, Z dimensions
    x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
    x_pred_fft = torch.fft.fftn(x_pred, dim=(-3, -2, -1))
    
    # Compute L2 distance of complex FFT coefficients
    # torch.abs computes magnitude of complex numbers
    fft_diff = x_fft - x_pred_fft
    fft_loss = torch.mean(torch.abs(fft_diff) ** 2)
    
    if normalize:
        # Normalize by number of spatial elements
        num_elements = x.shape[-3] * x.shape[-2] * x.shape[-1]
        fft_loss = fft_loss / num_elements
    
    return fft_loss


def get_eigloss(eigenvalues, threshold=1.0):
    """
    Eigenvalue regularisation for Stable Koopman.
    Penalises eigenvalue magnitudes that exceed *threshold* (unit circle).

    Args:
        eigenvalues: 1-D tensor of (real or complex) eigenvalues.
        threshold: Maximum allowed magnitude (default 1.0).

    Returns:
        Scalar loss: mean(max(0, |lambda_i| - threshold)^2).
    """
    if eigenvalues is None:
        return torch.tensor(0.0)
    if torch.is_complex(eigenvalues):
        mags = eigenvalues.abs()
    else:
        mags = eigenvalues.abs()
    excess = torch.clamp(mags - threshold, min=0.0)
    return torch.mean(excess ** 2)


def get_consistency_loss(transition_model, zt, dt, U):
    """
    Temporal consistency loss for Deep Koopman.
    Enforces that composing single-step operators equals a multi-step operator:
        ||K(z1) * z1 - z2||^2   where z1 = K(z0) * z0, z2 via two-step.

    Args:
        transition_model: The DeepKoopmanTransitionModel instance.
        zt: Initial latent state (batch, d).
        dt: Timestep tensor (batch, 1).
        U: List of control tensors for at least 2 steps.

    Returns:
        Scalar consistency loss.
    """
    if len(U) < 2:
        return torch.tensor(0.0, device=zt.device)
    z1, _ = transition_model.forward(zt, dt, U[0])
    z2_single, _ = transition_model.forward(z1, dt, U[1])
    Zt_k, _ = transition_model.forward_nsteps(zt, dt, U[:2])
    z2_multi = Zt_k[-1]
    return torch.mean((z2_single - z2_multi) ** 2)


def get_energy_conservation_loss(energies):
    """
    Energy conservation loss for Hamiltonian Neural ODE.
    Penalises the variance of H across trajectory steps.

    Args:
        energies: List of 1-D tensors [H(z_0), H(z_1), ..., H(z_K)].

    Returns:
        Scalar loss: mean variance of H across the trajectory.
    """
    if energies is None or len(energies) < 2:
        return torch.tensor(0.0)
    stacked = torch.stack(energies, dim=0)
    return torch.mean(torch.var(stacked, dim=0))


def get_jacobian_reg_loss(z, x, n_projections=1):
    """
    Jacobian regularization for contractive encoders.
    Approximates ||dz/dx||_F^2 via random projections (Hutchinson estimator).

    Requires x.requires_grad == True before the encoder forward pass.

    Args:
        z: Encoder output (batch, latent_dim).  Must be connected to x in the graph.
        x: Encoder input (batch, C, X, Y, Z).   Must have requires_grad=True.
        n_projections: Number of random projections (1 is usually sufficient).

    Returns:
        Scalar loss approximating the squared Frobenius norm of the Jacobian.
    """
    B = z.size(0)
    total = 0.0
    for _ in range(n_projections):
        v = torch.randn_like(z)
        Jv = torch.autograd.grad(
            outputs=z, inputs=x,
            grad_outputs=v,
            create_graph=True, retain_graph=True
        )[0]
        total = total + torch.mean(Jv.reshape(B, -1).pow(2).sum(dim=1))
    return total / n_projections


def get_cycle_consistency_loss(z_original, z_re_encoded):
    """
    Cycle-consistency loss: penalizes encode(decode(z)) != z.
    Makes the encoder idempotent under the decode-re-encode cycle.

    Args:
        z_original: Original latent encoding (batch, d).
        z_re_encoded: Re-encoded latent after decode-then-encode (batch, d).

    Returns:
        Scalar MSE loss between original and re-encoded latents.
    """
    return torch.mean((z_original - z_re_encoded) ** 2)


def get_masked_reconstruction_loss(x, x_pred, mask, reconstruction_variance=0.1):
    """
    Calculate reconstruction loss only on active (unmasked) cells.
    
    This loss excludes inactive reservoir cells (non-physical regions like
    boundaries, non-reservoir rock, etc.) from the reconstruction loss
    calculation, ensuring the model focuses on learning actual reservoir behavior.
    
    Args:
        x: True spatial state tensor [batch, channels, X, Y, Z]
        x_pred: Predicted/reconstructed spatial state tensor [batch, channels, X, Y, Z]
        mask: Active cell mask where True=active cell to include in loss
              Can be [X, Y, Z] for global mask or [batch, X, Y, Z] for case-specific
        reconstruction_variance: Noise variance for scaling (from config)
    
    Returns:
        Masked reconstruction loss normalized by active cell count
    """
    # Expand mask to match tensor dimensions [batch, channels, X, Y, Z]
    if mask.dim() == 3:
        # Global mask [X, Y, Z] -> [1, 1, X, Y, Z]
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 4:
        # Case-specific mask [batch, X, Y, Z] -> [batch, 1, X, Y, Z]
        mask_expanded = mask.unsqueeze(1)
    else:
        # Already 5D, use as-is
        mask_expanded = mask
    
    # Broadcast mask to all channels
    mask_expanded = mask_expanded.expand_as(x)
    
    # Compute squared error
    squared_error = (x - x_pred) ** 2
    
    # Apply mask (zero out inactive cells)
    masked_error = squared_error * mask_expanded.float()
    
    # Normalize by active cell count (not total cells)
    # This ensures loss scale is independent of how many cells are masked
    active_count = mask_expanded.sum()
    v = reconstruction_variance
    
    # Add small epsilon to prevent division by zero
    return masked_error.sum() / (2 * v * active_count + 1e-8)


def get_dissipativity_loss(singular_values, margin=0.1):
    """
    Dissipativity regularisation for Dissipative Koopman.
    Penalises singular values that are too close to 1 (the
    contractivity boundary), encouraging a safety margin.

    Args:
        singular_values: 1-D tensor of constrained singular values in [0, 1).
        margin: Target distance from 1. Singular values above (1-margin)
                incur a quadratic penalty.

    Returns:
        Scalar loss: mean(max(0, sigma_i - (1-margin))^2).
    """
    threshold = 1.0 - margin
    excess = torch.clamp(singular_values - threshold, min=0.0)
    return torch.mean(excess ** 2)


def get_reversibility_loss(residual):
    """
    Reversibility loss for IS-FNO transition.
    The residual is ||z - lift_inv(lift(z))||^2 averaged over the batch,
    already computed inside the transition model.

    Args:
        residual: Scalar tensor — mean squared reversibility error.

    Returns:
        The residual itself (pass-through for weighting by lambda).
    """
    return residual


def get_sindy_sparsity_loss(coefficients):
    """
    L1 sparsity penalty on SINDy coefficient matrix.
    Encourages discovery of sparse symbolic governing equations.

    Args:
        coefficients: [library_dim, d] parameter matrix.

    Returns:
        Scalar loss: mean absolute value of all coefficients.
    """
    return torch.mean(torch.abs(coefficients))


def get_sindy_consistency_loss(z_dot_model, z_dot_sindy):
    """
    Consistency loss between actual latent velocity and SINDy-predicted
    velocity.  Ensures the SINDy library faithfully represents the
    learned dynamics.

    Args:
        z_dot_model: [batch, d] — (z_{t+1} - z_t) / dt from the transition.
        z_dot_sindy: [batch, d] — Theta(z_t) @ xi from the SINDy library.

    Returns:
        Scalar loss: MSE between the two velocity estimates.
    """
    return torch.mean((z_dot_model - z_dot_sindy) ** 2)


def get_sde_kl_loss(diffusion_values):
    """
    KL-like regulariser for Latent SDE diffusion magnitude.
    Prevents the diffusion from collapsing to zero (no stochasticity)
    or exploding (unstable trajectories).  Penalises deviation of
    mean diffusion from a unit-scale prior.

    Args:
        diffusion_values: Scalar tensor — mean diffusion magnitude.

    Returns:
        Scalar loss: (diffusion - 1)^2 soft penalty.
    """
    return (diffusion_values - 1.0) ** 2
