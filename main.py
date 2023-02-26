import warnings
warnings.filterwarnings("ignore")
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import random
import time 
from datetime import timedelta

import hydra
import numpy as np
import torch
import wandb
from pathlib import Path
from dm_env import specs

# from dreamer import Dreamer 
from utils.env import make_env
import utils.helper as helper
from utils.video import VideoRecorder
from utils.logger import Logger
from utils.buffer import ReplayBufferStorage, make_replay_loader 

torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = "cfgs", "logs"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = Path.cwd() / __LOGS__ / cfg.algo_name / cfg.exp_name / cfg.env_name / str(cfg.seed) 

        if cfg.seed < 0: cfg.seed = random.randint(0, 10000) # generate random seed if cfg.seed<0 (-1 by default)
        set_seed(cfg.seed)

        # fill some cfg value on the fly #
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.episode_length = cfg.episode_length // cfg.action_repeat
        cfg.train_step = cfg.train_episode * cfg.episode_length
        
        self.device = torch.device(cfg.device)
        self.cfg = cfg
        
        self.setup()

        # dreamer_kwargs = {
            
        # }
        # self.agent = Dreamer(**dreamer_kwargs)

        self.timer = helper.Timer()

        self._global_step = 0
        self._global_episode = 0

    def setup(self,):
        # create envs
        self.train_env = make_env(self.cfg.env_name, self.cfg.seed, self.cfg.action_repeat,
                                       self.cfg.modality, self.cfg.frame_stack)
        self.eval_env = make_env(self.cfg.env_name, self.cfg.seed+100, self.cfg.action_repeat,
                                       self.cfg.modality, self.cfg.frame_stack)
        self.cfg.obs_shape = tuple(int(x) for x in self.train_env.observation_spec().shape)
        self.cfg.action_shape = tuple(int(x) for x in self.train_env.action_spec().shape)

        # create folders
        if self.cfg.save_model:
            self.model_dir = self.work_dir / 'models'
            helper.make_dir(self.model_dir) 
        if self.cfg.save_buffer:
            buffer_dir = self.work_dir / 'buffer'
            helper.make_dir(buffer_dir)
        if self.cfg.save_logging:
            # create logger
            logs_dir = self.work_dir/'logging'
            helper.make_dir(self.work_dir / "logging") 
            self.logger = Logger(logs_dir)
            # initialize wandb logging if needed # TODO: move it to logger
            if self.cfg.use_wandb:
                wandb.init(project="dreamer", name=f'{self.cfg.env_name}-{self.cfg.algo_name}-{self.cfg.exp_name}-{str(self.cfg.seed)}-{int(time.time())}',
                                        group=f'{self.cfg.env_name}-{self.cfg.algo_name}', 
                                        tags=[self.cfg.algo_name, self.cfg.env_name, self.cfg.exp_name, str(self.cfg.seed)],
                                        config=self.cfg,
                                        monitor_gym=True)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_buffer, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(self.work_dir) if self.cfg.save_video else None

    @property
    def global_step(self):
        return self._global_step
    
    @property
    def global_episode(self):
        return self._global_episode
    
    @property
    def global_frame(self,):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self,):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = helper.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), helper.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = helper.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = helper.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = helper.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), helper.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='default')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()    