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


class Dreamer(object):
    def __init__(self,device):
        self.device = torch.device(device)

        # models
        self.encoder = nets.encoder().to(self.device)
        self.decoder = nets.decoder().to(self.device)
        self.rssm = nets.RSSM().to(self.device)
        self.reward_fn = nets.mlp().to(self.device)
        self.value = nets.Value().to(self.device)
        self.value_tar = nets.ValueTar().to(self.device)
        self.actor = nets.Actor().to(self.device)

        # init optimizers
        self.world_param = chain(self.rssm.parameters(), self.encoder.parameters(),
                                self.decoder.parameters(), self.reward_fn.parameters())
        self.world_optim = optim.Adam(self.world_param, lr=world_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.value_optim = optim.Adam(self.value.parameters(), lr=value_lr)
    
        # free nats
        self.free_nats = torch.full((1,), free_nats, dtype=torch.float32, device=self.device)


    def infer_state(self, rssmState, action, observation):
        prior_rssmState, posterior_rssmState = self.rssm.onestep_observe(self.encoder(observation, rssmState, action))
        return posterior_rssmState 

    def _update_world_model(self, actions, rewards, nonterminals, obses):
        init_rssmState = self.rssm.init_rssmState()
        obs_embeddings = self.encoder(obses)
        prior_rssmState, posterior_rssmState = self.rssm.rollout(init_rssmState, actions, nonterminals, obs_embeddings)

        reconstruction_loss = F.mse_loss(self.decoder(posterior_rssmState), 
                                        obses, reduction='none').sum(dim=(2, 3, 4)).mean(dim=(0, 1))
        reward_loss = F.mse_loss(self.reward_fn(posterior_rssmState), rewards, reduction='none').mean(dim=(0, 1))
        kl_dyn = torch.max(
            kl_divergence(Independent(Normal())),
            self.free_nats
        )
        kl_rep = torch.max(
            kl_divergence(),
            self.free_nats
        )

        loss = reconstruction_loss + reward_loss + kl_dyn + kl_rep

        self.world_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_param, self.grad_clip_norm, norm_type=2)
        self.world_optim.step()

        return {'world_loss': loss.item()}

    def _update_actor(self):
        pass

    def _update_critic(self):
        pass
    
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

    def _image(self):
        pass

    def update(self, replay_iter, batch_size):
        
        metrics = {}
        metrics.update(self._update_world_loss())

        set_requires_grad(self.world_param, False)
        set_requires_grad(self.value.parameters(), False)
        
        # latent imagination
        imag_rssmStates, imag_logp, pi_entropy = self._image()
        metrics.update(self._update_actor())
        
        # update value function
        set_requires_grad(self.value.parameters(), True)
        metrics.update(self._update_actor())
        set_requires_grad(self.world_param, True)

        return metrics


    @torch.no_grad()
    def select_action(self, obs, eval_mode=False):
        pass

    def save(self, fp):
        pass

    def load(self, fp):
        pass