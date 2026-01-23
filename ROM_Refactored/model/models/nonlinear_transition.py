"""
Non-linear transition model for E2C architecture
Uses ODE-based dynamics in latent space
"""

import torch
import torch.nn as nn
from model.models.transition_utils import create_node_encoder, ode_solve


class NonLinearTransitionModel(nn.Module):
    def __init__(self, latent_dim, u_dim, nsteps):
        super(NonLinearTransitionModel, self).__init__()
        self.latent_dim = latent_dim
        self.latent_dim_total = latent_dim + u_dim
        self.u_dim = u_dim 
        self.node_encoder = create_node_encoder(self.latent_dim_total, self.u_dim)
        self.steps = nsteps
        # self.feature = ode
        self.norm = nn.BatchNorm2d(128)

    def forward(self, zt, dt, ut):
        hz = self.node_encoder
        # ut_dt = ut * dt
        zt1 = ode_solve(zt, ut, dt, self.steps, hz)
        # zt1 = torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
        
#         print('predicted latent shape', zt1.shape)
        return zt1

