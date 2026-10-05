import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, done_config="never", eval_mode=False, **kwargs):
        self.done_config = "never" if eval_mode else done_config
        self._velocity_threshold = 2.5
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    def constraint_value(self, ob, xvel):
        return np.min(
            [
                (ob[1] + 0.3) * (1.0 / 0.3),
                (0.3 - ob[1]) * (1.0 / 0.3),
                (self._velocity_threshold - xvel) * (1.0 / self._velocity_threshold),
            ]
        )

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        constraint_value_before = self.constraint_value(self._get_obs(), self.sim.data.qvel[0])

        self.do_simulation(action, self.frame_skip)

        xposafter = self.sim.data.qpos[0]
        xvelafter = self.sim.data.qvel[0]
        ob = self._get_obs()
        constraint_value_after = self.constraint_value(ob, xvelafter)

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        cost = float(constraint_value_after < 0.0)

        if self.done_config == "done_on_violation":
            notdone = np.isfinite(ob).all() and -0.3 <= ob[1] <= 0.3 and xvelafter <= self._velocity_threshold
            done = not notdone
        elif self.done_config == "never":
            done = False
        else:
            raise NotImplementedError(f"done_config {self.done_config} is not supported")

        return ob, reward, done, dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            constraint_value_before=constraint_value_before,
            constraint_value_after=constraint_value_after,
            cost=cost,
        )

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        for i in range(7):
            position[i + 2] = self.normalize_angle(position[i + 2])
        position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()


    @staticmethod
    def normalize_angle(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi
