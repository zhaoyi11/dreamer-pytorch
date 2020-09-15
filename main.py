import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from memory import ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel
from utils import lineplot, write_video, cal_returns
from tensorboardX import SummaryWriter


# Hyperparameters
parser = argparse.ArgumentParser(description='Dreamer')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS + CONTROL_SUITE_ENVS, help='Gym/Control Suite environment')
parser.add_argument('--symbolic', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--cnn-act', type=str, default='relu', choices=dir(F), help='Model activation function for a convolution layer')
parser.add_argument('--dense-act', type=str, default='elu', choices=dir(F), help='Model activation function a dense layer')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=300, metavar='H', help='Hidden size')  # paper:300; tf_implementation:400; aligned wit paper. 
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--worldmodel-LogProbLoss', action='store_true', help='use LogProb loss for observation_model and reward_model training')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--world_lr', type=float, default=6e-4, metavar='α', help='Learning rate') 
parser.add_argument('--actor_lr', type=float, default=8e-5, metavar='α', help='Learning rate') 
parser.add_argument('--value_lr', type=float, default=8e-5, metavar='α', help='Learning rate') 
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value') 
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
parser.add_argument('--expl_amount', type=float, default=0.3, help='exploration noise')
parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=25, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=10, metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
args = parser.parse_args()
args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))


# Setup
results_dir = os.path.join('results', args.env, str(args.seed))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
else:
  args.device = torch.device('cpu')

metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 
           'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': []}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.seed))

# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)

if args.experience_replay is not '' and os.path.exists(args.experience_replay):
  D = torch.load(args.experience_replay)
  metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
  D = ExperienceReplay(args.experience_size, args.symbolic, env.observation_size, env.action_size, args.bit_depth, args.device)
  # Initialise dataset D with S random seed episodes
  for s in range(1, args.seed_episodes + 1):
    observation, done, t = env.reset(), False, 0
    while not done:
      action = env.sample_random_action()
      next_observation, reward, done = env.step(action)
      D.append(observation, action, reward, done)
      observation = next_observation
      t += 1
    metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
    metrics['episodes'].append(s)


# Initialise model parameters randomly
transition_model = TransitionModel(
  args.belief_size, 
  args.state_size, 
  env.action_size, 
  args.hidden_size, 
  args.embedding_size, 
  args.dense_act).to(device=args.device)

observation_model = ObservationModel(
  args.symbolic, 
  env.observation_size, 
  args.belief_size, 
  args.state_size, 
  args.embedding_size, 
  activation_function=(args.dense_act if args.symbolic else args.cnn_act)).to(device=args.device)

reward_model = RewardModel(
  args.belief_size, 
  args.state_size, 
  args.hidden_size, 
  args.dense_act).to(device=args.device)

encoder = Encoder(
  args.symbolic, 
  env.observation_size, 
  args.embedding_size, 
  args.cnn_act).to(device=args.device)

actor_model = ActorModel(
  env.action_size, 
  args.belief_size, 
  args.state_size, 
  args.hidden_size, 
  activation_function=args.dense_act).to(device=args.device)

value_model = ValueModel(
  args.belief_size, 
  args.state_size, 
  args.hidden_size, 
  args.dense_act).to(device=args.device)

world_param = list(transition_model.parameters()) + list(observation_model.parameters())  + list(reward_model.parameters())  + list(encoder.parameters())

world_optimizer = optim.Adam(world_param, lr=args.world_lr)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=args.actor_lr)
value_optimizer = optim.Adam(value_model.parameters(), lr=args.value_lr)

if args.models is not '' and os.path.exists(args.models):
  model_dicts = torch.load(args.models)
  transition_model.load_state_dict(model_dicts['transition_model'])
  observation_model.load_state_dict(model_dicts['observation_model'])
  reward_model.load_state_dict(model_dicts['reward_model'])
  encoder.load_state_dict(model_dicts['encoder'])
  actor_model.load_state_dict(model_dicts['actor_model'])
  value_model.load_state_dict(model_dicts['value_model'])
  world_optimizer.load_state_dict(model_dicts['world_optimizer'])

free_nats = torch.full((1, ), args.free_nats, dtype=torch.float32, device=args.device)  # Allowed deviation in KL divergence

def update_belief_and_act(args, env, actor_model, transition_model, encoder, belief, posterior_state, action, observation, explore=False):
  # Infer belief over current state q(s_t|o≤t,a<t) from the history
  belief, _, _, _, posterior_state, _, _ = transition_model(
    posterior_state, 
    action.unsqueeze(dim=0), 
    belief, 
    encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
  
  belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
  
  if explore:
    action = actor_model(belief, posterior_state).rsample()  # batch_shape=1, event_shape=6
    # add exploration noise -- following the original code: line 275-280
    action = Normal(action, args.expl_amount).rsample()
    action = torch.clamp(action, -1, 1)
  else:
    action = actor_model(belief, posterior_state).mode()

  next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
  return belief, posterior_state, action, next_observation, reward, done


# Testing only
if args.test:
  # Set models to eval mode
  transition_model.eval()
  reward_model.eval()
  encoder.eval()
  with torch.no_grad():
    total_reward = 0
    for _ in tqdm(range(args.test_episodes)):
      observation = env.reset()

      belief = torch.zeros(1, args.belief_size, device=args.device)
      posterior_state = torch.zeros(1, args.state_size, device=args.device)
      action = torch.zeros(1, env.action_size, device=args.device)

      pbar = tqdm(range(args.max_episode_length // args.action_repeat))
      for t in pbar:
        belief, posterior_state, action, observation, reward, done = update_belief_and_act(
          args, 
          env, 
          actor_model, 
          transition_model, 
          encoder, belief, 
          posterior_state, action, 
          observation.to(device=args.device))

        total_reward += reward

        if args.render:
          env.render()
        if done:
          pbar.close()
          break

  print('Average Reward:', total_reward / args.test_episodes)
  env.close()
  quit()


# Training (and testing)
for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1):
  # Model fitting
  losses = []

  for s in tqdm(range(args.collect_interval)):
    # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
    observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size) # Transitions start at time t = 0
    
    """world model update"""
    init_belief = torch.zeros(args.batch_size, args.belief_size, device=args.device)
    init_state = torch.zeros(args.batch_size, args.state_size, device=args.device)
    # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(
      init_state, 
      actions[:-1], 
      init_belief, 
      bottle(encoder, (observations[1:], )),
      nonterminals[:-1])
   
    observation_loss = F.mse_loss(
      bottle(observation_model, (beliefs, posterior_states)),
      observations[1:], 
      reduction='none').sum(dim=2 if args.symbolic else (2, 3, 4)).mean(dim=(0, 1))


    reward_loss = F.mse_loss(
      bottle(reward_model, (beliefs, posterior_states)), 
      rewards[1:], 
      reduction='none').mean(dim=(0,1))
    
    # transition loss
    kl_loss = torch.max(
      kl_divergence(
        Independent(Normal(posterior_means, posterior_std_devs), 1), 
        Independent(Normal(prior_means, prior_std_devs),1)), 
      free_nats).mean(dim=(0,1)) 

    # Update model parameters
    world_optimizer.zero_grad()
    (observation_loss + reward_loss + kl_loss).backward()
    nn.utils.clip_grad_norm_(world_param, args.grad_clip_norm, norm_type=2)
    world_optimizer.step()

    """ actor model update """
    # freeze params to save memeory
    for p in world_param:
      p.requires_grad = False
    for p in value_model.parameters():
      p.requires_grad = False
    
    # Rollout to generate imagined trajectories
    C,B,_ = list(posterior_states.size())  # flatten the tensor
    flatten_size = C * B

    posterior_states = posterior_states.detach().reshape(flatten_size, -1)
    beliefs = beliefs.detach().reshape(flatten_size, -1)
  
    imag_beliefs, imag_states = [beliefs], [posterior_states]
    
    for i in range(args.planning_horizon):
      imag_action = actor_model(
        imag_beliefs[-1].detach(), 
        imag_states[-1].detach()).rsample().unsqueeze(dim=0)  # add the time dimension
      imag_belief, imag_state, _, _ = transition_model(imag_states[-1], imag_action, imag_beliefs[-1])
      imag_beliefs.append(imag_belief.squeeze(dim=0))
      imag_states.append(imag_state.squeeze(dim=0))

    imag_beliefs = torch.stack(imag_beliefs, dim=0).to(args.device)  # shape [horizon+1, (chuck-1)*batch, belief_size]
    imag_states = torch.stack(imag_states, dim=0).to(args.device)
    
    # reward and value prediction of imagined trajectories
    imag_reward = bottle(reward_model, (imag_beliefs, imag_states))
    imag_value = bottle(value_model, (imag_beliefs, imag_states))
    
    pcont = args.discount * torch.ones_like(imag_reward)

    returns = cal_returns(imag_reward[:-1], imag_value[:-1], imag_value[-1], pcont[:-1], lambda_=args.disclam)

    discount = torch.cumprod(torch.cat([torch.ones_like(pcont[:1]), pcont[:-2]], 0), 0) 
    discount = discount.detach()

    assert list(discount.size()) == list(returns.size())
    actor_loss = -torch.mean(discount * returns)
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip_norm, norm_type=2)
    actor_optimizer.step()

    for p in world_param:
      p.requires_grad = True
    for p in value_model.parameters():
      p.requires_grad = True

    """ critic model update """
    imag_beliefs = imag_beliefs.detach()
    imag_states = imag_states.detach()
    target_return = returns.detach()

    value_pred = bottle(value_model, (imag_beliefs, imag_states))[:-1]
    value_loss = F.mse_loss(value_pred, target_return, reduction="none").mean(dim=(0,1))

    value_optimizer.zero_grad()
    value_loss.backward()
    nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm, norm_type=2)
    value_optimizer.step()

    losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item(), actor_loss.item(), value_loss.item()])


  # Update and plot loss metrics
  losses = tuple(zip(*losses))
  metrics['observation_loss'].append(losses[0])
  metrics['reward_loss'].append(losses[1])
  metrics['kl_loss'].append(losses[2])
  metrics['actor_loss'].append(losses[3])
  metrics['value_loss'].append(losses[4])
  lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['actor_loss']):], metrics['actor_loss'], 'actor_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)


  # Data collection
  with torch.no_grad():
    observation, total_reward = env.reset(), 0

    belief = torch.zeros(1, args.belief_size, device=args.device)
    posterior_state = torch.zeros(1, args.state_size, device=args.device)
    action = torch.zeros(1, env.action_size, device=args.device)

    pbar = tqdm(range(args.max_episode_length // args.action_repeat))
    for t in pbar:
      belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
        args, 
        env, 
        actor_model, 
        transition_model, 
        encoder, 
        belief, 
        posterior_state, 
        action, 
        observation.to(device=args.device), 
        explore=True)

      D.append(observation, action.cpu(), reward, done)
      total_reward += reward
      observation = next_observation
      if args.render:
        env.render()
      if done:
        pbar.close()
        break
    
    # Update and plot train reward metrics
    metrics['steps'].append(t + metrics['steps'][-1])
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)
    lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir)


  # Test model
  if episode % args.test_interval == 0:
    # Set models to eval mode
    transition_model.eval()
    observation_model.eval()
    reward_model.eval() 
    encoder.eval()
    actor_model.eval()
    value_model.eval()

    # Initialise parallelised test environments
    test_envs = EnvBatcher(
      Env, 
      (args.env, args.symbolic, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth), 
      {}, 
      args.test_episodes)
    
    with torch.no_grad():
      observation = test_envs.reset()
      total_rewards = np.zeros((args.test_episodes, ))
      video_frames = []

      belief = torch.zeros(args.test_episodes, args.belief_size, device=args.device) 
      posterior_state = torch.zeros(args.test_episodes, args.state_size, device=args.device)
      action = torch.zeros(args.test_episodes, env.action_size, device=args.device)
      
      for t in tqdm(range(args.max_episode_length // args.action_repeat)):
        belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
          args, 
          test_envs, 
          actor_model, 
          transition_model, 
          encoder, 
          belief, 
          posterior_state, 
          action, 
          observation.to(device=args.device))

        total_rewards += reward.numpy()
        if not args.symbolic:  # Collect real vs. predicted frames for video
          video_frames.append(make_grid(torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
        observation = next_observation
        if done.sum().item() == args.test_episodes:
          pbar.close()
          break
    
    # Update and plot reward metrics (and write video if applicable) and save metrics
    metrics['test_episodes'].append(episode)
    metrics['test_rewards'].append(total_rewards.tolist())
    lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
    lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'], 'test_rewards_steps', results_dir, xaxis='step')
    if not args.symbolic:
      episode_str = str(episode).zfill(len(str(args.episodes)))
      write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
      save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
    torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

    # Set models to train mode
    transition_model.train()
    observation_model.train()
    reward_model.train()
    encoder.train()
    actor_model.train()
    value_model.train()
    # Close test environments
    test_envs.close()

  writer.add_scalar("train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
  writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1], metrics['steps'][-1]*args.action_repeat)
  writer.add_scalar("observation_loss", metrics['observation_loss'][0][-1], metrics['steps'][-1])
  writer.add_scalar("reward_loss", metrics['reward_loss'][0][-1], metrics['steps'][-1])
  writer.add_scalar("kl_loss", metrics['kl_loss'][0][-1], metrics['steps'][-1])
  writer.add_scalar("actor_loss", metrics['actor_loss'][0][-1], metrics['steps'][-1])
  writer.add_scalar("value_loss", metrics['value_loss'][0][-1], metrics['steps'][-1])  
  print("episodes: {}, total_steps: {}, train_reward: {} ".format(metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1]))

  # Checkpoint models
  if episode % args.checkpoint_interval == 0:
    torch.save({'transition_model': transition_model.state_dict(),
                'observation_model': observation_model.state_dict(),
                'reward_model': reward_model.state_dict(),
                'encoder': encoder.state_dict(),
                'actor_model': actor_model.state_dict(),
                'value_model': value_model.state_dict(),
                'world_optimizer': world_optimizer.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'value_optimizer': value_optimizer.state_dict()
                }, os.path.join(results_dir, 'models_%d.pth' % episode))
    if args.checkpoint_experience:
      torch.save(D, os.path.join(results_dir, 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes


# Close training environment
env.close()
