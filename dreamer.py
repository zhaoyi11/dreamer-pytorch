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

import nets
import utils.helper as helper

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
                device, 
                mppi_kwargs=None):
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
        self.discount = 0.99
        self.disclam = 0.95

        self.mppi_kwargs = mppi_kwargs
        self.action_dim = action_dim

    def _update_world_model(self, obses, actions, rewards, nonterminals):
        """ The inputs sequences are: a, r, o | a, r, o| a, r, o"""
        L, B, x_dim = obses.shape
        init_rssmState = self.rssm.init_rssmState(B).to(device=self.device)
        obs_embeddings = self.encoder(obses) # TODO: might a bug here.
        prior_rssmState, posterior_rssmState = self.rssm.rollout(init_rssmState, actions, nonterminals, obs_embeddings)
        
        # TODO: might be a bug in the self.decoder() -- the input has one additional dim
        reconstruction_loss = F.mse_loss(self.decoder(posterior_rssmState.state), 
                                        obses, reduction='none').sum(dim=2).mean(dim=(0, 1))
        reward_loss = F.mse_loss(self.reward_fn(posterior_rssmState.state), rewards, reduction='none').sum(dim=2).mean(dim=(0, 1))
        
        # kl_dyn = torch.max(
        #     kl_divergence(prior_rssmState.detach().dist, posterior_rssmState.dist),
        #     self.free_nats).mean()
        
        # kl_rep = torch.max(
        #     kl_divergence(prior_rssmState.dist, posterior_rssmState.detach().dist),
        #     self.free_nats).mean()

        # TODO: check kl, detach() 
        kl_dyn = torch.max(kl_divergence(prior_rssmState.dist, posterior_rssmState.dist).sum(-1), self.free_nats).mean()
        kl_rep = 0.
        loss = reconstruction_loss + reward_loss + kl_dyn + kl_rep

        self.world_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_param, self.grad_clip_norm, norm_type=2)
        self.world_optim.step()

        return {'world_loss': loss.item(), }, posterior_rssmState

    def _update_actor_critic(self, rssmState, logp):

        set_requires_grad(self.value.parameters(), False)
        states = rssmState.state
        rewards = self.reward_fn(states)
        values = self.value_tar(states)
        
        pcont = self.discount * torch.ones_like(rewards).detach()

        # TODO: add logp
        # values[1:] -= 1e-5 * logp

        returns = self._cal_returns(rewards[:-1], values[:-1], values[-1], pcont[:-1], lambda_=self.disclam)
        discount = torch.cumprod(torch.cat([torch.ones_like(pcont[:1]),\
                                            pcont[:-2]],0),0).detach()
        actor_loss = -torch.mean(discount * returns)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()        

        set_requires_grad(self.value.parameters(), True)
        
        # update value function
        target_v = returns.detach()
        pred_v = self.value(states.detach())[:-1]
        value_loss = F.mse_loss(pred_v, target_v)

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
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
        imag_rssmStates, imag_logps = [rssmState], []
        for i in range(self.imag_length):
            _rssm_state = imag_rssmStates[-1]
            pi_dist = self.actor(_rssm_state.state) # TODO: decide the api
            action = pi_dist.rsample()
            imag_rssmStates.append(self.rssm.onestep_image(_rssm_state, action, nonterminal=True))
            imag_logps.append(pi_dist.log_prob(action).sum(-1, keepdim=True))
        import ipdb; ipdb.set_trace()
        # returned shape rssm_state: [imag_L+1, B, x_dim], logps: [imag_L, B, 1]
        return self.rssm.stack_rssmState(imag_rssmStates), torch.stack(imag_logps, dim=0).to(self.device)

    def update(self, replay_iter):
        batch = next(replay_iter)
        next_obs, action, reward, nonterminal = to_torch(batch, self.device, dtype=torch.float32)
        # swap the batch and horizon dimension -> [H, B, _shape]
        next_obs, action, reward, nonterminal = torch.swapaxes(next_obs, 0, 1), torch.swapaxes(action, 0, 1),\
                                                torch.swapaxes(reward, 0, 1),\
                                                torch.swapaxes(nonterminal, 0, 1)

        metrics = {}
        world_metrics, rssmState = self._update_world_model(next_obs, action, reward, nonterminal)
        metrics.update(world_metrics)

        set_requires_grad(self.world_param, False)
        # latent imagination
        imag_rssmStates, imag_logp = self._image(rssmState.detach().flatten())
        
        metrics.update(self._update_actor_critic(imag_rssmStates, imag_logp))
        set_requires_grad(self.world_param, True)

        # soft update the target value function
        helper.soft_update_params(self.value, self.value_tar, tau=0.005) # TODO: check whether we should tune the tau
        return metrics

    @torch.no_grad()
    def infer_state(self, rssmState, action, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        prior_rssmState, posterior_rssmState = self.rssm.onestep_observe(self.encoder(observation), rssmState, action)
        return posterior_rssmState 

    @torch.no_grad()
    def select_action(self, rssm_state, step, eval_mode=False):
        act_dist = self.actor(rssm_state.state)
        if not eval_mode:
            action = act_dist.sample()
            action += 0.3 * torch.rand_like(action)
        else:
            action = act_dist.mean

        return action[0]

    @torch.no_grad()
    def plan(self, rstate, step, eval_mode=False):
        # num_samples = self.mppi_kwargs.get('num_samples')
        # plan_horizon = self.mppi_kwargs.get('plan_horizon')
        # num_topk = self.mppi_kwargs.get('num_topk')
        # iteration = self.mppi_kwargs.get('iteration')
        # temp = self.mppi_kwargs.get('temperature')
        # momentum = self.mppi_kwargs.get('momentum')

        # TODO: add to config
        num_samples = 1000
        plan_horizon = 12
        num_topk = 100
        iteration = 10
        temp = 0.5
        momentum = 0.1
        action_noise=0.3
        
        rstate = rstate.repeat(num_samples, 1) # shape [num_samples, x_dim]
    
        mu = torch.zeros(plan_horizon, self.action_dim, device=self.device)
        std = torch.ones_like(mu)

        for _ in range(iteration):
            actions = mu.unsqueeze(1) + std.unsqueeze(1) * \
                    torch.randn(plan_horizon, num_samples, self.action_dim).to(device=self.device, dtype=mu.dtype)
            actions.clamp_(-1, 1) # shape: [plan_horizon, num_samples, action_dim]
            nonterminals = torch.ones(plan_horizon, num_samples, 1, device=self.device, dtype=mu.dtype)
            rstate_prior = self.rssm.rollout(rstate, actions, nonterminals)

            returns = self.reward_fn(rstate_prior.state)
            returns = returns.sum(dim=0).squeeze(-1)

            # Re-fit belief to the K best action sequences
            elite_idxs = torch.topk(returns, num_topk, dim=0, sorted=False).indices
            elite_returns, elite_actions = returns[elite_idxs], actions[:, elite_idxs]

            # update action_mean and action_std
            max_return = torch.max(returns)

            score = torch.exp(temp * (elite_returns - max_return))
            score /= score.sum()

            _mean = torch.sum(score.reshape(1, -1, 1) * elite_actions, dim=1) # weighted man and std over elites
            _stddev = torch.sqrt(torch.sum(score.reshape(1, -1, 1) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1))
            
            # momentum udate of mean
            mu, std = momentum * mu + (1. - momentum) * _mean, _stddev

        # outputs (weighted samples)
        score = score.cpu().numpy()
        output = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)] 

        if not eval_mode:
            output = output + action_noise * torch.randn_like(output)
        return output[0] # [action_dim]


    def save(self, fp):
        pass

    def load(self, fp):
        pass
