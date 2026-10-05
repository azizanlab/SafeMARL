import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, done_config="never", eval_mode=False, **kwargs):
        self.done_config = "never" if eval_mode else done_config
        self._velocity_threshold = 1.5
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def constraint_value(self, ob, xvel):
        return np.min(
            [
                ob[0] - 1.0,
                (1.8 - ob[0]) * (1.0 / 1.8),
                (self._velocity_threshold - xvel) * (1.0 / self._velocity_threshold),
            ]
        )

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        constraint_value_before = self.constraint_value(self._get_obs(), self.sim.data.qvel[0])

        self.do_simulation(a, self.frame_skip)

        posafter = self.sim.data.qpos[0]
        xvelafter = self.sim.data.qvel[0]
        ob = self._get_obs()
        constraint_value_after = self.constraint_value(ob, xvelafter)

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        cost = float(constraint_value_after < 0.0)

        if self.done_config == "done_on_violation":
            notdone = np.isfinite(ob).all() and 1.0 <= ob[0] <= 1.8 and xvelafter <= self._velocity_threshold
            done = not notdone
        elif self.done_config == "never":
            done = False
        else:
            raise NotImplementedError(f"done_config {self.done_config} is not supported")

        return ob, reward, done, dict(
            constraint_value_before=constraint_value_before,
            constraint_value_after=constraint_value_after,
            cost=cost,
        )

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        for i in range(7):
            qpos[i + 2] = self.normalize_angle(qpos[i + 2])
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()


    @staticmethod
    def normalize_angle(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi
