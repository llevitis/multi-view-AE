import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .layers import Encoder, Decoder 
from .utils_deep import Optimisation_DVCCA
import numpy as np

class DVCCA(nn.Module, Optimisation_DVCCA):

    def __init__(self, input_dims, config, private):
        '''
        Initialise the Deep Variational Canonical Correlation Analysis model

        input_dims: The input data dimension.
        config: Configuration dictionary.
        private: Label to indicate VCCA or VCCA-private.
        beta: KL weight.

        '''

        super().__init__()
        self._config = config
        self.model_type = 'DVCCA'
        if private:
            self.model_type = 'DVCCA_private'
        self.input_dims = input_dims
        self.hidden_layer_dims = config['hidden_layers']
        self.hidden_layer_dims.append(config['latent_size'])
        self.private = private
        self.n_views = len(input_dims)
        self.beta = config['beta']
        self.learning_rate = config['learning_rate']
        self.encoder = torch.nn.ModuleList([Encoder(input_dim = self.input_dims[0], hidden_layer_dims=self.hidden_layer_dims, variational=True)])
        if private:
            self.private_encoders = torch.nn.ModuleList([Encoder(input_dim = input_dim, hidden_layer_dims=self.hidden_layer_dims, variational=True) for input_dim in self.input_dims])
            self.hidden_layer_dims[-1] = config['latent_size'] + config['latent_size']

        self.decoders = torch.nn.ModuleList([Decoder(input_dim = input_dim, hidden_layer_dims=self.hidden_layer_dims) for input_dim in self.input_dims])
        if private:
            self.optimizers = [torch.optim.Adam(self.encoder.parameters(),lr=0.001)] + [torch.optim.Adam(list(self.decoders[i].parameters()),
                                      lr=self.learning_rate) for i in range(self.n_views)]
        else:
            self.optimizers = [torch.optim.Adam(self.encoder.parameters(), lr=0.001)] + [torch.optim.Adam(list(self.decoders[i].parameters()),
                                      lr=0.001) for i in range(self.n_views)]
    def encode(self, x):
        mu, logvar = self.encoder[0](x[0])
        if self.private:
            mu_tmp = []
            logvar_tmp = []
            for i in range(self.n_views):
                mu_p, logvar_p = self.private_encoders[i](x[i])
                mu_ = torch.cat((mu, mu_p),1)
                mu_tmp.append(mu_)
                logvar_ = torch.cat((logvar, logvar_p),1)
                logvar_tmp.append(logvar_)
            mu = mu_tmp
            logvar = logvar_tmp
        return mu, logvar
    
    def reparameterise(self, mu, logvar):
        if self.private:
            z = []
            for i in range(self.n_views):
                std = torch.exp(0.5*logvar[i])
                eps = torch.randn_like(mu[i])
                z.append(mu[i]+eps*std)
        else:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(mu)
            z = mu+eps*std
        return z

    def decode(self, z):
        x_recon = []
        for i in range(self.n_views):
            if self.private:
                x_out = self.decoders[i](z[i])
            else:
                x_out = self.decoders[i](z)
            x_recon.append(x_out)
        return x_recon

    def forward(self, x):
        self.zero_grad()
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z)
        fwd_rtn = {'x_recon': x_recon,
                    'mu': mu,
                    'logvar': logvar}
        return fwd_rtn

    @staticmethod
    def calc_kl(self, mu, logvar):
        '''
        Implementation from: https://arxiv.org/abs/1312.6114

        '''
        kl = 0
        for i in range(self.n_views):
            kl+= -0.5*torch.sum(1 + logvar[i] - mu[i].pow(2) - logvar[i].exp(), dim=-1).mean(0)
        return self.beta*kl/self.n_views

    @staticmethod
    def recon_loss(self, x, x_recon):
        recon_loss = 0
        for i in range(self.n_views):
            recon_loss+= torch.mean(((x_recon[i] - x[i])**2).sum(dim=-1))
        return recon_loss/self.n_views

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn['x_recon']
        mu = fwd_rtn['mu']
        logvar = fwd_rtn['logvar']

        kl = self.calc_kl(self, mu, logvar)
        recon = self.recon_loss(self, x, x_recon)

        total = kl + recon
        
        losses = {'total': total,
                'kl': kl,
                'reconstruction': recon}
        return losses


__all__ = [
    'DVCCA'
]