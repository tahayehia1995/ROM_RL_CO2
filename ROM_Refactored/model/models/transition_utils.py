"""
Utility functions for transition models
Includes encoder creation and ODE solving functions
"""

import torch
import torch.nn as nn
from model.layers.standard_layers import fc_bn_relu


def create_trans_encoder(total_input_dim, hidden_dims=[200, 200]):
    """
    Create transition encoder network
    
    Args:
        total_input_dim: Total input dimension (latent_dim + 1 for dt)
        hidden_dims: List of hidden layer dimensions (default: [200, 200])
        
    Returns:
        Sequential encoder network
    """
    layers = []
    prev_dim = total_input_dim
    
    # Handle None or missing hidden_dims by using default
    if hidden_dims is None:
        hidden_dims = [200, 200]
    
    # Add hidden layers
    for hidden_dim in hidden_dims:
        layers.append(fc_bn_relu(prev_dim, hidden_dim))
        prev_dim = hidden_dim
    
    # Add output layer
    layers.append(fc_bn_relu(prev_dim, total_input_dim - 1))
    
    trans_encoder = nn.Sequential(*layers)
    return trans_encoder


def create_nltrans_encoder(total_input_dim, u_dim):
    """
    Create non-linear transition encoder network
    
    Args:
        total_input_dim: Total input dimension (latent_dim + u_dim)
        u_dim: Control input dimension
        
    Returns:
        Sequential encoder network
    """
    trans_encoder = nn.Sequential(
        fc_bn_relu(total_input_dim, 200),
        fc_bn_relu(200, 200),
        fc_bn_relu(200, total_input_dim - u_dim)
    )
    return trans_encoder


class create_node_encoder(nn.Module):
    """Node encoder for ODE-based transition models"""
    def __init__(self, total_input_dim, u_dim):
        super(create_node_encoder, self).__init__()
        self.NNODE = create_nltrans_encoder(total_input_dim, u_dim)
        
    def forward(self, x, u):
        zt_expand = torch.cat([x, u], dim=-1)
        out = self.NNODE(zt_expand)
        return out


def ode_solve(z0, ut, dt, nsteps, func):
    """
    Solve ODE using Euler method
    
    Args:
        z0: Initial latent state
        ut: Control input
        dt: Time step
        nsteps: Number of integration steps
        func: ODE function (node encoder)
        
    Returns:
        Final latent state after integration
    """
    n_steps = nsteps
    z = z0
    for i_step in range(n_steps):
        z = z + dt/n_steps * func(z, ut)
    return z

