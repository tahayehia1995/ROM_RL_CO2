"""
Linear transition model for E2C architecture
Uses linear state-space dynamics in latent space
"""

import torch
import torch.nn as nn
from model.models.transition_utils import create_trans_encoder
from model.utils.initialization import weights_init


class LinearTransitionModel(nn.Module):
    def __init__(self, config):
        super(LinearTransitionModel, self).__init__()
        self.config = config
        self.latent_dim = config['model']['latent_dim']
        self.u_dim = config['model']['u_dim']
        self.num_prob = config['data']['num_prod']
        self.num_inj = config['data']['num_inj']
        self.trans_encoder = create_trans_encoder(
            self.latent_dim + 1, 
            config['transition']['encoder_hidden_dims']
        )
        self.trans_encoder.apply(weights_init)
        
        self.At_layer = nn.Linear(self.latent_dim, self.latent_dim * self.latent_dim)
        self.At_layer.apply(weights_init)
        self.Bt_layer = nn.Linear(self.latent_dim, self.latent_dim * self.u_dim)
        self.Bt_layer.apply(weights_init)
        
        self.Ct_layer = nn.Linear(self.latent_dim, (self.num_prob*2+ self.num_inj)* self.latent_dim)
        self.Ct_layer.apply(weights_init)
        self.Dt_layer = nn.Linear(self.latent_dim, (self.num_prob*2+ self.num_inj) * self.u_dim)
        self.Dt_layer.apply(weights_init)

    def forward_nsteps(self, zt, dt, U):
#         print(dt.shape)
#         print(zt.shape)
        zt_expand = torch.cat([zt, dt], dim=-1)

        hz = self.trans_encoder(zt_expand)
#         print(hz.shape)
        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.u_dim)

        Zt_k =[]
        Yt_k = []
        for ut in U:
            ut_dt = ut * dt
            zt = torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
            
            yt = torch.bmm(Ct, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1)
            
            Zt_k.append(zt)
            Yt_k.append(yt)
        # print('predicted At shape', At.shape)
        # return Zt_k, gershgorin_loss(At), gershgorin_loss(Bt)
        return Zt_k, Yt_k

    def forward(self, zt, dt, ut):
#         print(dt.shape)
#         print(zt.shape)
        zt_expand = torch.cat([zt, dt], dim=-1)

        hz = self.trans_encoder(zt_expand)
#         print(hz.shape)
        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)

        Ct = self.Ct_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.u_dim)
        
        ut_dt = ut * dt

        zt1 = torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
        yt1 = torch.bmm(Ct, zt1.unsqueeze(-1)).squeeze(-1) + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1)
            
#         print('predicted latent shape', zt1.shape)
        return zt1, yt1

class LinearMultiTransitionModel(nn.Module):
    def __init__(self, latent_dim, u_dim, num_prob, num_inj, nsteps):
        super(LinearMultiTransitionModel, self).__init__()
        self.latent_dim = latent_dim
        self.u_dim = u_dim 
        self.num_prob = num_prob 
        self.num_inj = num_inj
        self.nsteps = nsteps
        self.trans_encoder = create_trans_encoder(self.latent_dim + 1)
        self.trans_encoder.apply(weights_init)
        
        self.At_layer = nn.Linear(self.latent_dim, self.latent_dim * self.latent_dim)
        self.At_layer.apply(weights_init)
        self.Bt_layer = nn.Linear(self.latent_dim, self.latent_dim * self.u_dim)
        self.Bt_layer.apply(weights_init)
        
        self.Ct_layer = nn.Linear(self.latent_dim, (self.num_prob*2+ self.num_inj)* self.latent_dim)
        self.Ct_layer.apply(weights_init)
        self.Dt_layer = nn.Linear(self.latent_dim, (self.num_prob*2+ self.num_inj) * self.u_dim)
        self.Dt_layer.apply(weights_init)

    def forward_nsteps(self, zt, dt, U):
#         print(dt.shape)
#         print(zt.shape)
        zt_expand = torch.cat([zt, dt], dim=-1)

        hz = self.trans_encoder(zt_expand)
#         print(hz.shape)
        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.u_dim)

        Zt_k =[]
        Yt_k = []
        for ut in U:
            ut_dt = ut * dt
            zt = torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
            
            yt = torch.bmm(Ct, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1)
            
            Zt_k.append(zt)
            Yt_k.append(yt)
        # print('predicted At shape', At.shape)
        # return Zt_k, gershgorin_loss(At), gershgorin_loss(Bt)
        return Zt_k, Yt_k

    def forward(self, zt, dt, ut):
#         print(dt.shape)
#         print(zt.shape)
        zt_expand = torch.cat([zt, dt], dim=-1)

        hz = self.trans_encoder(zt_expand)
#         print(hz.shape)
        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)

        Ct = self.Ct_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.u_dim)
        
        ut_dt = ut * dt

        zt1 = torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
        yt1 = torch.bmm(Ct, zt1.unsqueeze(-1)).squeeze(-1) + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1)
            
#         print('predicted latent shape', zt1.shape)
        return zt1, yt1

