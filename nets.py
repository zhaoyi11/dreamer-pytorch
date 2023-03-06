from typing import Optional, List
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td
from torch.distributions.normal import Normal
from torch.distributions.transforms import Transform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np

import utils.helper as h

class RSSMState():
    def __init__(self, deter, stoc_mean, stoc_std):
        stoc = stoc_mean + stoc_std * torch.rand_like(stoc_std)
        state = torch.cat([deter, stoc], dim=-1)
        self.field = {'deter': deter, 'stoc_mean': stoc_mean, 
                    'stoc_std': stoc_std, 'stoc': stoc, 'state': state}

    def detach(self,):
        for k, v in self.field.items():
            self.field[k] = v.detach()
        return self

    def flatten(self,):
        "The returned shape is [B*N, x_dim]."
        for k, v in self.field.items():
            self.field[k] = v.reshape(-1, v.shape[-1])
        return self
    
    def suqeeze(self, dim=0):
        for k, v in self.field.items():
            self.field[k] = v.squeeze(dim)
        return self

    def unsqueeze(self, dim=0):
        for k, v in self.field.items():
            self.field[k] = v.unsqueeze(dim)
        return self

    def repeat(self, *size):
        for k, v in self.field.items():
            self.field[k] = v.repeat(*size)
        return self
    
    def to(self, device='cpu', dtype=torch.float32):
        for k, v in self.field.items():
            self.field[k] = v.to(device=device, dtype=dtype)
        return self

    @property
    def deter(self):
        return self.field['deter']

    @property
    def stoc(self):
        return self.field['stoc']
    
    @property
    def state(self):
        return self.field['state']
    
    @property
    def dist(self):
        mean, std = self.field['stoc_mean'], self.field['stoc_std']
        return td.independent.Independent(Normal(mean, std), 1)


class RSSM(nn.Module):
    def __init__(self, deter_dim, stoc_dim, embedding_dim, action_dim, mlp_dim, act_fn=nn.ELU, min_std_dev=0.1):
        super().__init__()
        self.min_std_dev = min_std_dev
        self.deter_dim, self.stoc_dim, self.embedding_dim = deter_dim, stoc_dim, embedding_dim

        self.fc_input = nn.Sequential(
            nn.Linear(stoc_dim + action_dim, deter_dim), act_fn()
        )
        
        self.rnn = nn.GRUCell(deter_dim, deter_dim)

        self.fc_prior = nn.Sequential(
            nn.Linear(deter_dim, mlp_dim), act_fn(),
            nn.Linear(mlp_dim, 2 * stoc_dim)
        )

        self.fc_posterior = nn.Sequential(
            nn.Linear(deter_dim + embedding_dim, mlp_dim), act_fn(),
            nn.Linear(mlp_dim, 2 * stoc_dim)
        )


    def onestep_image(self, rssmState, action, nonterminal=True):
        _input = self.fc_input(torch.cat([rssmState.stoc * nonterminal, action], dim=-1))
        deter_state = self.rnn(_input, rssmState.deter * nonterminal)

        prior_mu, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
        prior_std = F.softplus(prior_std) + self.min_std_dev

        prior_rssmState = RSSMState(deter_state, prior_mu, prior_std)

        return prior_rssmState

    def onestep_observe(self, obs_embedding, rssmState, action, nonterminal=True):
        prior_rssmState = self.onestep_image(rssmState, action, nonterminal)
        deter_state = prior_rssmState.deter
        
        _input = torch.cat([deter_state, obs_embedding], dim=-1)

        posterior_mu, posterior_std = torch.chunk(self.fc_posterior(_input), 2, dim=-1)
        posterior_std = F.softplus(posterior_std) + self.min_std_dev

        posterior_rssmState = RSSMState(deter_state, posterior_mu, posterior_std)

        return prior_rssmState, posterior_rssmState
    
    def rollout(self, init_rssmState, actions, nonterminals, obs_embeddings=None):
        """ The inputs/outputs sequences are
            time  : 0 1 2 3
            rssmS : x
            obs   : - x x x (optional)
            action: x x x 
            nonter: x x x
            output: - x x x
        """
        prior = []
        if obs_embeddings is not None:
            posterior = []

        rssmState = init_rssmState
        for t, (action, nonterminal) in enumerate(zip(actions, nonterminals)):
            if obs_embeddings is not None:
                prior_rssmState, posterior_rssmState = self.onestep_observe(obs_embeddings[t], rssmState, action, nonterminal)
                prior.append(prior_rssmState)
                posterior.append(posterior_rssmState)
            else:
                prior_rssmState = self.onestep_image(rssmState, action, nonterminal)
                prior.append(prior_rssmState)
        
        if obs_embeddings is not None:
            return self.stack_rssmState(prior), self.stack_rssmState(posterior)
        else:
            return self.stack_rssmState(prior)
    
    def stack_rssmState(self, state_list):
        rssmState = state_list[0]
        keys = rssmState.field.keys() # get keys of the RSSMState
        data = {k: [] for k in keys}
        # fill data
        for state in state_list:
            for k in keys:
                data[k].append(state.field[k])
        
        # set data to rssmState        
        for k in keys:
            rssmState.field[k] = torch.stack(data[k], dim=0)
        return rssmState

    def init_rssmState(self, length=1):
        return RSSMState(torch.zeros(length, self.deter_dim), 
                        torch.zeros(length, self.stoc_dim),
                        torch.ones(length, self.stoc_dim))

def mlp(in_dim, mlp_dims: List[int], out_dim, act_fn=nn.ELU, out_act=nn.Identity):
    """Returns an MLP."""
    if isinstance(mlp_dims, int): raise ValueError("mlp dimensions should be list, but got int.")

    layers = [nn.Linear(in_dim, mlp_dims[0]), act_fn()]
    for i in range(len(mlp_dims)-1):
        layers += [nn.Linear(mlp_dims[i], mlp_dims[i+1]), act_fn()]

    layers += [nn.Linear(mlp_dims[-1], out_dim), out_act()]
    return nn.Sequential(*layers)


class NormalizeImg(nn.Module):
    """Normalizes pixel observations to [0,1) range."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.)


class Flatten(nn.Module):
    """Flattens its input to a (batched) vector."""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)

def encoder():
    """Returns a TOLD encoder."""
    def _get_out_shape(in_shape, layers):
        """Utility function. Returns the output shape of a network for a given input shape."""
        x = torch.randn(*in_shape).unsqueeze(0)
        return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape

    if cfg.modality == 'pixels':
        C = int(3*cfg.frame_stack)
        layers = [NormalizeImg(),
                nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.ReLU(),
                nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.ReLU(),
                nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
                nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU()]
        out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
        layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
    else:
        layers = [nn.Linear(cfg.obs_shape[0], cfg.enc_dim), nn.ELU(),
                nn.Linear(cfg.enc_dim, cfg.latent_dim)]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, deter_dim, stoc_dim, mlp_dims, action_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(deter_dim+stoc_dim, mlp_dims[0]),
                            nn.LayerNorm(mlp_dims[0]), nn.Tanh())
        self._actor = mlp(mlp_dims[0], mlp_dims[1:], action_dim * 2)
        self.mu_scale = 5
        self.init_std = 5
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)
        self.min_std=1e-4
        self.apply(orthogonal_init) 

    def forward(self, obs):
        feature = self.trunk(obs)
        x = self._actor(feature)
        mu, std = torch.chunk(x, 2, dim=-1)
        # bound the action to [-mu_scale, mu_scale] --> to avoid numerical instabilities.  For computing log-probabilities, we need to invert the tanh and this becomes difficult in highly saturated regions.
        mu = self.mu_scale * torch.tanh(mu / self.mu_scale) 
        std = F.softplus(std + self.raw_init_std) + self.min_std
        return h.SquashedNormal(mu, std)
    

class Value(nn.Module):
    def __init__(self, deter_dim, stoc_dim, mlp_dims):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(deter_dim+stoc_dim, mlp_dims[0]),
                            nn.LayerNorm(mlp_dims[0]), nn.Tanh())
        self._value = mlp(mlp_dims[0], mlp_dims[1:], 1)

        self.apply(orthogonal_init)

    def forward(self, z):
        feature = self.trunk(z)
        return self._value(feature)


class CNNEncoder(nn.Module):
    def __init__(self, obs_shape, mlp_dims, latent_dim):
        super().__init__()
        self._encoder = net.mlp(obs_shape[0], mlp_dims, latent_dim,)
        self.apply(net.orthogonal_init)

    def forward(self, obs):
        out = self._encoder(obs)
        return out


class CNNDecoder(nn.Module):
    pass


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear): 
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
