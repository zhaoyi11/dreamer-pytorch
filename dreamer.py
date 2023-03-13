import os
from copy import deepcopy
from itertools import chain

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler # float16
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
                algo_name,
                deter_dim, stoc_dim, mlp_dim, embedding_dim,
                obs_shape, action_dim, 
                mlp_layer,
                world_lr, actor_lr, value_lr, grad_clip_norm, weight_decay,
                actor_ent, free_nats,
                coef_pred, coef_dyn, coef_rep,
                imag_length,
                device, 
                target_update_freq=100,
                num_channels=32,
                mppi_kwargs=None):
        self.device = torch.device(device)
        self.modality = modality
        # models
        if modality == 'pixels':
            self.encoder = nets.CNNEncoder(obs_shape, num_channels, embedding_dim).to(self.device)
            self.decoder = nets.CNNDecoder(deter_dim+stoc_dim, num_channels, obs_shape).to(self.device)
            pass
        else:
            self.encoder = nets.MLP(obs_shape[0], [mlp_dim]*mlp_layer, embedding_dim).to(self.device)
            self.decoder = nets.MLP(deter_dim+stoc_dim, [mlp_dim]*mlp_layer, obs_shape[0]).to(self.device)

        self.rssm = nets.RSSM(deter_dim, stoc_dim, embedding_dim, action_dim, mlp_dim).to(self.device)
        self.reward_fn = nets.MLP(deter_dim+stoc_dim, [mlp_dim]*mlp_layer, 1).to(self.device)

        self.value = nets.MLP(deter_dim+stoc_dim, [mlp_dim]*mlp_layer, 1).to(self.device)
        self.value_tar = nets.MLP(deter_dim+stoc_dim, [mlp_dim]*mlp_layer, 1).to(self.device)
        for p in self.value_tar.parameters():
            p.requires_grad = False

        self.actor = nets.Actor(deter_dim, stoc_dim, [mlp_dim]*mlp_layer, action_dim).to(self.device)

        # init optimizers
        self.world_param = chain(self.rssm.parameters(), self.encoder.parameters(),
                                self.decoder.parameters(), self.reward_fn.parameters())
        self.world_optim = optim.Adam(self.world_param, lr=world_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.value_optim = optim.Adam(self.value.parameters(), lr=value_lr)
    
        # grad scaler
        self.world_scaler = GradScaler()
        self.actor_scaler = GradScaler()
        self.value_scaler = GradScaler()

        self.free_nats = torch.tensor(free_nats, dtype=torch.float32, device=self.device)

        self.coef_pred, self.coef_dyn, self.coef_rep = coef_pred, coef_dyn, coef_rep
        self.grad_clip_norm = grad_clip_norm
        self.imag_length = imag_length
        self.discount = 0.99
        self.disclam = 0.95

        self.mppi_kwargs = mppi_kwargs
        self.action_dim = action_dim

        self.actor_ent = actor_ent
        self.algo_name = algo_name
        self.target_update_freq = target_update_freq
        self.update_counter = 0

    def _update_world_model(self, obses, actions, rewards, nonterminals):
        """ The inputs sequences are: a, r, o | a, r, o| a, r, o,
        obses in shape [H, B, *obs_dim]"""
        B = obses.shape[1]
        init_rstate = self.rssm.init_rstate(B).to(device=self.device)
        
        with autocast(dtype=torch.float16):
            obs_embeddings = self.encoder(obses) 
            prior_rstate, pos_rstate = self.rssm.rollout(init_rstate, actions, nonterminals, obs_embeddings)
            
            _obs_dim = list(range(obses.ndim)[2:]) # if modality is state, it's 2, and if modality == pixels, it's (2, 3, 4)
            rec_loss = F.mse_loss(self.decoder(pos_rstate.state), 
                                            obses, reduction='none').sum(dim=_obs_dim).mean(dim=(0, 1))
            reward_loss = F.mse_loss(self.reward_fn(pos_rstate.state[:-1]),
                                    rewards[1:], reduction='none').sum(dim=2).mean(dim=(0, 1)) # (s1, a1, r1, d1, s2) --> r1 = f(s1)  
            
            kl_rep = torch.maximum(kl_divergence(prior_rstate.dist, pos_rstate.dist).mean(), self.free_nats)
            kl_dyn = 0.
            # kl_rep = torch.maximum(
            #     kl_divergence(pos_rstate.dist, prior_rstate.detach().dist).mean(),
            #     self.free_nats) 

            # kl_dyn = torch.maximum(
            #     kl_divergence(pos_rstate.detach().dist, prior_rstate.dist).mean(),
            #     self.free_nats)
            loss = rec_loss + reward_loss + 0.2 * kl_rep + 0.8 * kl_dyn

        self.world_optim.zero_grad(set_to_none=True)
        self.world_scaler.scale(loss).backward()
        self.world_scaler.unscale_(self.world_optim)
        grad_norm = nn.utils.clip_grad_norm_(self.world_param, self.grad_clip_norm, norm_type=2)
        self.world_scaler.step(self.world_optim)
        self.world_scaler.update()

        return {'world_loss': loss.item(),
                'rec_loss': rec_loss.item(), 
                'reward_loss': reward_loss.item(),
                # 'kl_dyn_loss': kl_dyn.item(),
                'kl_rep_loss': kl_rep.item(),
                'prior_ent': prior_rstate.dist.entropy().mean().item(),
                'posterior_ent': pos_rstate.dist.entropy().mean().item(),
                'state_mean': pos_rstate.state.mean().item(),
                'state_max': pos_rstate.state.max().item(),
                'state_min': pos_rstate.state.min().item(),
                'world_grad_norm': grad_norm.item()},\
                pos_rstate

    def _update_actor_critic(self, imag_rstate, policy_ent):
        set_requires_grad(self.value.parameters(), False)
        
        with autocast(dtype=torch.float16):
            states = imag_rstate.state
            rewards = self.reward_fn(states)
            values = self.value_tar(states)
            
            pcont = self.discount * torch.ones_like(rewards).detach()
            returns = self._cal_returns(rewards[:-1], values[:-1], values[-1], pcont[:-1], lambda_=self.disclam)
            weights = torch.cumprod(torch.cat([torch.ones_like(pcont[:1]),\
                                                pcont[:-2]],0),0).detach()        
            
            # update actor
            # Actions:      0   [a1]  [a2]   a3
            #                  ^  |  ^  |  ^  |
            #                 /   v /   v /   v
            # States:     [z0]->[z1]-> z2 -> z3
            # Targets:     t0   [t1]  [t2]
            # Entropies:        [e1]  [e2]   e3
            # Weights:    [ 1]  [w1]   w2    
            # Loss:              l1    l2

            objective = returns[1:] # drop the first returns at the start of the traj because it from the replay buffer
            objective += self.actor_ent * policy_ent[:-1].unsqueeze(-1)
            
            actor_loss = -torch.mean(weights[:-1] * objective)
        
        self.actor_optim.zero_grad(set_to_none=True)
        self.actor_scaler.scale(actor_loss).backward()
        self.actor_scaler.step(self.actor_optim)      
        self.actor_scaler.update()

        # update value function
        # States:     [z0]  [z1]  [z2]   z3
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]   v3
        # Weights:    [ 1]  [w1]  [w2]   w3
        # Targets:    [t0]  [t1]  [t2]
        # Loss:        l0    l1    l2
        set_requires_grad(self.value.parameters(), True)
        with autocast(dtype=torch.float16):
            target_v = returns.detach()
            pred_v = self.value(states.detach())[:-1]
            value_loss = F.mse_loss(pred_v, target_v)

        self.value_optim.zero_grad(set_to_none=True)
        self.value_scaler.scale(value_loss).backward()
        self.value_scaler.step(self.value_optim)
        self.value_scaler.update()
        return {'value': pred_v.mean().item(), 
                'actor_loss':actor_loss.item(), 
                'value_loss':value_loss.item(),
                'policy_ent': policy_ent.mean().item(),}
    
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

    @autocast(dtype=torch.float16)
    def _image(self, rstate):
        # rollout the dynamic model forward to generate imagined trajectories 
        imag_rstates, imag_logps = [rstate], []
        for i in range(self.imag_length):
            _rstate = imag_rstates[-1]
            pi_dist = self.actor(_rstate.state.detach()) # notice that the state is detached
            action = pi_dist.rsample()
            imag_rstates.append(self.rssm.image_step(_rstate, 
                                                    action, nonterminal=True))
            imag_logps.append(pi_dist.log_prob(action)) # TODO: check wehter to detach action
        # returned shape rstate: [imag_L+1, B, x_dim], logps: [imag_L, B] # TODO: be careful of the dimension.
        # - before logp is to convert it to entropy
        return self.rssm.stack_rstate(imag_rstates), -torch.stack(imag_logps, dim=0)

    def update(self, replay_iter):
        self.update_counter += 1
        batch = next(replay_iter)
        next_obs, action, reward, nonterminal = to_torch(batch, self.device, dtype=torch.float32)
        # swap the batch and horizon dimension -> [H, B, _shape]
        next_obs, action, reward, nonterminal = torch.swapaxes(next_obs, 0, 1), torch.swapaxes(action, 0, 1),\
                                                torch.swapaxes(reward, 0, 1),\
                                                torch.swapaxes(nonterminal, 0, 1)

        if self.modality == 'pixels':
            next_obs = helper.norm_pixels(next_obs)

        metrics = {}
        world_metrics, rstate = self._update_world_model(next_obs, action, reward, nonterminal)
        metrics.update(world_metrics)


        # update actor critic in dreamer
        if self.algo_name in ['dreamerv1', 'dreamerv2']:
            set_requires_grad(self.world_param, False)
            # latent imagination
            imag_rstates, policy_ent = self._image(rstate.detach().flatten())
            
            metrics.update(self._update_actor_critic(imag_rstates, policy_ent))
            set_requires_grad(self.world_param, True)
        
            if self.update_counter % self.target_update_freq == 0:
                # update the target value function
                helper.soft_update_params(self.value, self.value_tar, tau=1)
        return metrics

    @torch.no_grad()
    def infer_state(self, prev_rstate, action, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.modality == 'pixels':
                helper.norm_pixels(observation)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        _, pos_rstate = self.rssm.obs_step(self.encoder(observation), prev_rstate, action)
        return pos_rstate 

    @torch.no_grad()
    def select_action(self, rstate, step, eval_mode=False):
        act_dist = self.actor(rstate.state)
        if not eval_mode:
            action = act_dist.sample()
            action += 0.2 * torch.rand_like(action)
        else:
            action = act_dist.mean

        return action[0]
    
    @torch.no_grad()
    def reset(self):
        rstate, action = self.rssm.init_rstate().to(device=self.device),\
                             np.zeros(self.action_dim) # init the dummy rstate and action
        return rstate, action

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

            _mean = torch.sum(score.reshape(1, -1, 1) * elite_actions, dim=1) # weighted mean and std over elites
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
