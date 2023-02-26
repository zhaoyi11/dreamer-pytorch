import imageio
import numpy as np


class VideoRecorder:
    """Utility class for logging evaluation videos."""
    def __init__(self, root_dir, wandb=None, render_size=256, fps=20): 
        self.save_dir = (root_dir / 'eval_video') if root_dir else None
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self._wandb = wandb
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size, width=self.render_size, camera_id=0)
            else:
                frame = env.render(mode='rgb_array', height=self.render_size, width=self.render_size, camera_id=0)
            self.frames.append(frame)

    def save(self, step):
        if self.enabled:
            frames = np.stack(self.frames).transpose(0, 3, 1, 2)
            path = self.save_dir / f"{step}.mp4"
            imageio.mimsave(str(path), self.frames, fps=self.fps)
