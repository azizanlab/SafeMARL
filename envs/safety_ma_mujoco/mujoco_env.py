import os

import gym
import mujoco_py
import numpy as np
from gym import spaces
from gym.utils import seeding


class MujocoEnv(gym.Env):
    def __init__(self, model_path, frame_skip):
        fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)

        self.observation_space = spaces.Box(
            np.full(observation.shape, -float("inf"), dtype=np.float32),
            np.full(observation.shape, float("inf"), dtype=np.float32),
            dtype=observation.dtype,
        )

        self.seed()

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_model(self):
        raise NotImplementedError

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def close(self):
        pass

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
