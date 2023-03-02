import os
from copy import deepcopy
from itertools import chain

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from tqdm import tqdm
# from memory import ExperienceReplay
# from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel, PCONTModel
import nets

def set_requires_grad(param, value):
	"""Enable/disable gradients for a given (sub)network."""
	for p in param:
		p.requires_grad_(value)


def count_vars(module):
  """ count parameters number of module"""
  return sum([np.prod(p.shape) for p in module.parameters()])


def to_torch(xs, device, dtype=torch.float32):
    return tuple(torch.as_tensor(x, device=device, dtype=dtype) for x in xs)

class Dreamer(object):
    def __init__(self,
                modality,
                deter_dim, stoc_dim, mlp_dim, embedding_dim,
                obs_shape, action_dim, 
                world_lr, actor_lr, value_lr, grad_clip_norm,
                free_nats,
                coef_pred, coef_dyn, coef_rep,
                imag_length,
                device):
        self.device = torch.device(device)

        # models
        if modality == 'pixels':
            pass
            # self.encoder = nets.CNNEncoder().to(self.device)
            # self.decoder = nets.CNNDecoder().to(self.device)
        else:
            self.encoder = nets.mlp(obs_shape[0], [mlp_dim, mlp_dim], embedding_dim).to(self.device)
            self.decoder = nets.mlp(deter_dim+stoc_dim, [mlp_dim, mlp_dim], obs_shape[0]).to(self.device)

        self.rssm = nets.RSSM(deter_dim, stoc_dim, embedding_dim, action_dim, mlp_dim).to(self.device)
        self.reward_fn = nets.mlp(deter_dim+stoc_dim, [mlp_dim, mlp_dim], 1).to(self.device)

        self.value = nets.Value(deter_dim, stoc_dim, mlp_dims=[mlp_dim, mlp_dim]).to(self.device)
        self.value_tar = nets.Value(deter_dim, stoc_dim, mlp_dims=[mlp_dim, mlp_dim]).to(self.device)

        self.actor = nets.Actor(deter_dim, stoc_dim, [mlp_dim, mlp_dim], action_dim).to(self.device)

        # init optimizers
        self.world_param = chain(self.rssm.parameters(), self.encoder.parameters(),
                                self.decoder.parameters(), self.reward_fn.parameters())
        self.world_optim = optim.Adam(self.world_param, lr=world_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.value_optim = optim.Adam(self.value.parameters(), lr=value_lr)
    
        # free nats
        self.free_nats = torch.full((1,), free_nats, dtype=torch.float32, device=self.device)

        self.coef_pred, self.coef_dyn, self.coef_rep = coef_pred, coef_dyn, coef_rep
        self.grad_clip_norm = grad_clip_norm
        self.imag_length = imag_length

    def infer_state(self, rssmState, action, observation):
        prior_rssmState, posterior_rssmState = self.rssm.onestep_observe(self.encoder(observation), rssmState, action)
        return posterior_rssmState 

    def _update_world_model(self, obses, actions, rewards, nonterminals):
        """ The inputs sequences are: a, r, o | a, r, o| a, r, o"""
        L, B, x_dim = obses.shape
        init_rssmState = self.rssm.init_rssmState(L).to(device=self.device)
        obs_embeddings = self.encoder(obses) # TODO: might a bug here.
        prior_rssmState, posterior_rssmState = self.rssm.rollout(init_rssmState, actions, nonterminals, obs_embeddings)
        
        # TODO: might be a bug in the self.decoder() -- the input has one additional dim
        reconstruction_loss = F.mse_loss(self.decoder(posterior_rssmState.state), 
                                        obses, reduction='none').sum(dim=2).mean(dim=(0, 1))
        reward_loss = F.mse_loss(self.reward_fn(posterior_rssmState.state), rewards, reduction='none').sum(dim=2).mean(dim=(0, 1))
        
        kl_dyn = torch.max(
            kl_divergence(prior_rssmState.detach().dist, posterior_rssmState.dist),
            self.free_nats).mean()
        
        kl_rep = torch.max(
            kl_divergence(prior_rssmState.dist, posterior_rssmState.detach().dist),
            self.free_nats).mean()

        loss = reconstruction_loss + reward_loss + kl_dyn + kl_rep

        self.world_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_param, self.grad_clip_norm, norm_type=2)
        self.world_optim.step()

        return {'world_loss': loss.item()}, posterior_rssmState

    def _update_actor(self, rssmState, logp):
        
        return {}
    
    def _update_critic(self, rssmState, logp):
        return {}
    
    def _cal_returns(self, reward, value, bootstrap, pcont, lambda_):
        """
        Calculate the target value, following equation (5-6) in Dreamer
        :param reward, value: imagined rewards and values, dim=[horizon, (chuck-1)*batch, reward/value_shape]
        :param bootstrap: the last predicted value, dim=[(chuck-1)*batch, 1(value_dim)]
        :param pcont: gamma
        :param lambda_: lambda
        :return: the target value, dim=[horizon, (chuck-1)*batch, value_shape]
        """
        assert list(reward.shape) == list(value.shape), "The shape of reward and value should be similar"
        if isinstance(pcont, (int, float)):
            pcont = pcont * torch.ones_like(reward)

        next_value = torch.cat((value[1:], bootstrap[None]), 0)  # bootstrap[None] is used to extend additional dim
        inputs = reward + pcont * next_value * (1 - lambda_)  # dim=[horizon, (chuck-1)*B, 1]
        outputs = []
        last = bootstrap

        for t in reversed(range(reward.shape[0])): # for t in horizon
            inp = inputs[t]
            last = inp + pcont[t] * lambda_ * last
            outputs.append(last)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns

    def _image(self, rssmState):
        # rollout the dynamic model forward to generate imagined trajectories 
        # TODO: check states.requires_grad
        imag_rssmStates, imag_logps = [rssmState], []
        for i in range(self.imag_length):
            _rssm_state = imag_rssmStates[-1]
            pi_dist = self.actor(_rssm_state.state) # TODO: decide the api
            action = pi_dist.rsample()
            imag_rssmStates.append(self.rssm.onestep_image(_rssm_state, action, nonterminal=True))
            imag_logps.append(pi_dist.log_prob(action).sum(-1, keepdim=True))

        return self.rssm.stack_rssmState(imag_rssmStates), torch.stack(imag_logps, dim=0).to(self.device)


    def update(self, replay_iter):
        batch = next(replay_iter)
        obs, action, reward, nonterminal = to_torch(batch, self.device, dtype=torch.float32)
        # swap the batch and horizon dimension -> [H, B, _shape]
        obs, action, reward, nonterminal = torch.swapaxes(obs, 0, 1), torch.swapaxes(action, 0, 1),\
                                                torch.swapaxes(reward, 0, 1),\
                                                torch.swapaxes(nonterminal, 0, 1)

        metrics = {}
        world_metrics, rssmState = self._update_world_model(obs, action, reward, nonterminal)
        metrics.update()

        set_requires_grad(self.world_param, False)
        set_requires_grad(self.value.parameters(), False)
        
        # latent imagination
        imag_rssmStates, imag_logp = self._image(rssmState.detach().flatten())
        import ipdb; ipdb.set_trace()
        metrics.update(self._update_actor(imag_rssmStates, imag_logp))
        
        # update value function
        set_requires_grad(self.value.parameters(), True)
        metrics.update(self._update_critic(imag_rssmStates.detach(), imag_logp))
        set_requires_grad(self.world_param, True)

        return metrics


    @torch.no_grad()
    def select_action(self, obs, step, eval_mode=False):
        pass

    def save(self, fp):
        pass

    def load(self, fp):
        pass