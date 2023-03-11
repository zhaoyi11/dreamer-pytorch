from typing import Optional, List
import copy
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
    def __init__(self, deter, stoc_mean, stoc_std, stoc=None, state=None):
        if stoc is None:
            stoc = stoc_mean + stoc_std * torch.rand_like(stoc_std)
        if state is None:
            state = torch.cat([deter, stoc], dim=-1)
        
        self.field = {'deter': deter, 'stoc_mean': stoc_mean, 
                        'stoc_std': stoc_std, 'stoc': stoc, 'state': state}

    def detach(self,):
        field = {}
        for k, v in self.field.items():
            field[k] = v.detach()
        return RSSMState(**field)

    def flatten(self,):
        "The returned shape is [B*N, x_dim]."
        field = {}
        for k, v in self.field.items():
            field[k] = v.reshape(-1, v.shape[-1])
        return RSSMState(**field)
    
    def suqeeze(self, dim=0):
        field = {}
        for k, v in self.field.items():
            field[k] = v.squeeze(dim)
        return RSSMState(**field)

    def unsqueeze(self, dim=0):
        field = {}
        for k, v in self.field.items():
            self.field[k] = v.unsqueeze(dim)
        return RSSMState(**field)

    def repeat(self, *size):
        field = {}
        for k, v in self.field.items():
            field[k] = v.repeat(*size)
        return RSSMState(**field)
    
    def to(self, device='cpu', dtype=torch.float32):
        field = {}
        for k, v in self.field.items():
            field[k] = v.to(device=device, dtype=dtype)
        return RSSMState(**field)

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
        # the event_shape equals the last dimension, and the rest will be the batch shape
        # for example, the mean has shape [20, 30, 50], then the even_shape is [50], and the batch_shape is [20, 30]
        return td.independent.Independent(Normal(mean, std), 1)

class RSSM(nn.Module):
    def __init__(self, deter_dim, stoc_dim, embedding_dim, action_dim, mlp_dim, act_fn=nn.ELU, min_std_dev=0.1):
        super().__init__()
        self.min_std_dev = min_std_dev
        self.deter_dim, self.stoc_dim, self.embedding_dim = deter_dim, stoc_dim, embedding_dim

        self.fc_input = nn.Sequential(
            nn.Linear(stoc_dim + action_dim, deter_dim), act_fn())
        
        self.rnn = nn.GRUCell(deter_dim, deter_dim)

        self.fc_prior = nn.Sequential(
            nn.Linear(deter_dim, mlp_dim), act_fn(),
            nn.Linear(mlp_dim, 2 * stoc_dim))

        self.fc_posterior = nn.Sequential(
            nn.Linear(deter_dim + embedding_dim, mlp_dim), act_fn(),
            nn.Linear(mlp_dim, 2 * stoc_dim))

    def image_step(self, prev_rstate, action, nonterminal=True):
        """
            [rstate] -> prior_rstate
            [action] /
        """
        _input = self.fc_input(torch.cat([prev_rstate.stoc * nonterminal, action], dim=-1))
        deter_state = self.rnn(_input, prev_rstate.deter)

        prior_mu, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
        prior_std = F.softplus(prior_std) + self.min_std_dev

        prior_rstate = RSSMState(deter_state, prior_mu, prior_std)
        return prior_rstate
    
    def obs_step(self, obs_embedding, prev_rstate, action, nonterminal=True):
        prior_rstate = self.image_step(prev_rstate, action, nonterminal)
        deter_state = prior_rstate.deter
        
        _input = torch.cat([deter_state, obs_embedding], dim=-1)

        pos_mu, pos_std = torch.chunk(self.fc_posterior(_input), 2, dim=-1)
        pos_std = F.softplus(pos_std) + self.min_std_dev

        pos_rstate = RSSMState(deter_state, pos_mu, pos_std)
        return prior_rstate, pos_rstate
    
    def rollout(self, init_rstate, actions, nonterminals, obs_embeddings=None):
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

        rstate = init_rstate
        for t, (action, nonterminal) in enumerate(zip(actions, nonterminals)):
            if obs_embeddings is not None:
                prior_rstate, pos_rstate = self.obs_step(obs_embeddings[t], rstate, action, nonterminal)
                prior.append(prior_rstate)
                posterior.append(pos_rstate)
    
                # update rstate
                rstate = pos_rstate
            else:
                prior_rstate = self.image_step(rstate, action, nonterminal)
                prior.append(prior_rstate)

                # update rstate
                rstate = prior_rstate
        
        if obs_embeddings is not None:
            return self.stack_rstate(prior), self.stack_rstate(posterior)
        else:
            return self.stack_rstate(prior)
    
    def stack_rstate(self, rstate_list):
        rstate = rstate_list[0]
        keys = rstate.field.keys() # get keys of the RSSMState
        _data = {k: [] for k in keys}
        field = {}
        # fill data
        for k in keys:
            for state in rstate_list:
                _data[k].append(state.field[k])
            field[k] = torch.stack(_data[k], dim=0)
        return RSSMState(**field)

    def init_rstate(self, batch_size=1):
        return RSSMState(torch.zeros(batch_size, self.deter_dim, dtype=torch.float32), 
                        torch.zeros(batch_size, self.stoc_dim, dtype=torch.float32),
                        torch.zeros(batch_size, self.stoc_dim, dtype=torch.float32))


def mlp(in_dim, mlp_dims: List[int], out_dim, act_fn=nn.ELU, out_act=nn.Identity):
    """Returns an MLP."""
    if isinstance(mlp_dims, int): raise ValueError("mlp dimensions should be list, but got int.")

    layers = [nn.Linear(in_dim, mlp_dims[0]), act_fn()]
    for i in range(len(mlp_dims)-1):
        layers += [nn.Linear(mlp_dims[i], mlp_dims[i+1]), act_fn()]

    layers += [nn.Linear(mlp_dims[-1], out_dim), out_act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, deter_dim, stoc_dim, mlp_dims, action_dim):
        super().__init__()
        self._actor = mlp(deter_dim + stoc_dim, mlp_dims, action_dim * 2)
        self.mu_scale = 5
        self.init_std = 5
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)
        self.min_std = 1e-4
        # self.apply(orthogonal_init) 

    def forward(self, obs):
        x = self._actor(obs)
        mu, std = torch.chunk(x, 2, dim=-1)
        # bound the action to [-mu_scale, mu_scale] --> to avoid numerical instabilities.  
        # For computing log-probabilities, we need to invert the tanh and this becomes difficult in highly saturated regions.
        mu = self.mu_scale * torch.tanh(mu / self.mu_scale) 
        std = F.softplus(std + self.raw_init_std) + self.min_std
        return td.independent.Independent(h.SquashedNormal(mu, std), 1)


class Flatten(nn.Module):
    """Flattens its input to a (batched) vector."""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNEncoder(nn.Module):
    def __init__(self, obs_shape, num_channels, embedding_dim):
        super().__init__()
        C, H, W = obs_shape
        C = int(C) # num of input channels
        assert H == W == 64 # the architecture is for images with shape C*64*64

        _layers = [
            nn.Conv2d(C, 8 * num_channels, 4, stride=2), nn.ReLU(),
            nn.Conv2d(8 * num_channels, 4 * num_channels, 4, stride=2), nn.ReLU(),
            nn.Conv2d(4 * num_channels, 2 * num_channels, 4, stride=2), nn.ReLU(),
            nn.Conv2d(2 * num_channels, num_channels, 4, stride=2), nn.ReLU()]
        output_shape = self._get_output_shape((C, H, W), _layers)
        if np.prod(output_shape) == embedding_dim:
            _layers.extend([nn.Identity()])
        else:
            _layers.extend([Flatten(), nn.Linear(np.prod(output_shape), embedding_dim)])

        self._encoder = nn.Sequential(*_layers)

        # self.apply(orthogonal_init)

    def _get_output_shape(self, in_shape, layers):
        """Utility function. Returns the output shape of a network for a given input shape."""
        x = torch.randn(*in_shape).unsqueeze(0)
        return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape

    def forward(self, obs):
        batch_shape = obs.shape[:-3] 
        img_shape = obs.shape[-3:]
        out = self._encoder(obs.reshape(-1, *img_shape))
        out = torch.reshape(out, (*batch_shape, -1))
        return out


class CNNDecoder(nn.Module):
    def __init__(self, input_dim, num_channels, output_shape):
        super().__init__()
        self.embedding_dim = 32 * num_channels
        self.output_shape = output_shape
        C = int(output_shape[0]) # input channel
        self._fc1 = nn.Linear(input_dim, self.embedding_dim)
        self._decoder = nn.Sequential(nn.ConvTranspose2d(self.embedding_dim, 4 * num_channels, 5, stride=2), nn.ReLU(),
                            nn.ConvTranspose2d(4 * num_channels, 2 * num_channels, 5, stride=2), nn.ReLU(),
                            nn.ConvTranspose2d(2 * num_channels, num_channels, 6, stride=2), nn.ReLU(),
                            nn.ConvTranspose2d(num_channels, C, 6, stride=2))

        # self.apply(orthogonal_init)

    def forward(self, x):
        batch_shape = x.shape[:-1]
        input_dim = x.shape[-1]
        squeezed_dim = np.prod(batch_shape).item()
        x = x.reshape(squeezed_dim, input_dim)
        x = self._fc1(x)
        x = x.reshape(squeezed_dim, self.embedding_dim, 1, 1)
        x = self._decoder(x)
        out = x.reshape(*batch_shape, *self.output_shape)
        return out


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
