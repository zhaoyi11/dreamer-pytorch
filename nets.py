from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions.transforms import Transform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np


# @dataclass
# class RSSMState():
#     def __init__(self,):
#         deter:
#         stoc:
#         stoc_mean:
#         stoc_std:
#         state:         

#     def detach(self,):
#         pass

#     def sample(self,):
#         pass
    
#     def flatten(self,):
#         "The returned shape is [B*N, x_dim]."
#         pass
    



class RSSM(nn.Module):
    def __init__(self, deter_dim, stoc_dim, embedding_dim, action_dim, mlp_dim, act_fn=nn.ELU, min_std_dev=0.1):
        super().__init__()
        self.min_std_dev = min_std_dev
        
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

        prior_rssmState = RSSMState(prior_mu, prior_std, deter_state)

        return prior_rssmState

    def onestep_observe(self, obs_embedding, rssmState, action, nonterminal=True):
        prior_rssmState = self.onestep_image(rssmState, action, nonterminal)
        deter_state = prior_rssmState.deter
        
        _input = torch.cat([deter_state, obs_embedding], dim=-1)

        posterior_mu, posterior_std = torch.chunk(self.fc_posterior(_input), 2, dim=-1)
        posterior_std = F.softplus(posterior_std) + self.min_std_dev

        posterior_rssmState = RSSMState(posterior_mu, posterior_std, deter_state)

        return prior_rssmState, posterior_rssmState
    
    def rollout(self, init_rssmState, actions, nonterminals, obs_embeddings=None):
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
            return self._stack_rssmState(prior), self._stack_rssmState(posterior)
        else:
            return self._stack_rssmState(prior)
    
    def _stack_rssmState(self, states):
        pass


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
        self._actor = mlp(mlp_dims[0], mlp_dims[1:], action_dim)
        self.apply(orthogonal_init) 

    def forward(self, obs, std):
        feature = self.trunk(obs)
        mu = self._actor(feature)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return h.TruncatedNormal(mu, std)
    

class Value(nn.Module):
    def __init__(self, deter_dim, stoc_dim, mlp_dims):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(deter_dim+stoc_dim, mlp_dims[0]),
                            nn.LayerNorm(mlp_dims[0]), nn.Tanh())
        self._v1 = mlp(mlp_dims[0], mlp_dims[1:], 1)
        self._v2 = mlp(mlp_dims[0], mlp_dims[1:], 1)
        self.apply(orthogonal_init)

    def forward(self, z):
        feature = self.trunk(z)
        return self._v1(feature), self._v2(feature)


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
