import mujoco_py as mjp
import numpy as np
from gym import utils

from safety_ma_mujoco import mujoco_env

_TAN30 = np.tan(30 / 360 * 2 * np.pi)


def wall_distances(x, y):
    ywall = np.array([-5.0, 5.0])
    if x < 20:
        return y - x * _TAN30 + ywall
    elif 20 < x < 60:
        return y + (x - 40) * _TAN30 - ywall
    elif 60 < x < 100:
        return y - (x - 80) * _TAN30 + ywall
    else:
        return y - 20 * _TAN30 + ywall


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, done_config="never", eval_mode=False, **kwargs):
        self.done_config = "never" if eval_mode else done_config
        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _torso_rot_z(self):
        q = self.data.get_body_xquat("torso")
        return 1 - 2 * (q[1] ** 2 + q[2] ** 2)

    def constraint_value(self):
        xpos, ypos = self.get_body_com("torso")[0], self.get_body_com("torso")[1]
        d = wall_distances(xpos, ypos)
        z = self.state_vector()[2]
        return np.min(
            [
                (abs(d[0]) - 1.8) * (1.0 / 1.8),
                (abs(d[1]) - 1.8) * (1.0 / 1.8),
                (z - 0.2) * (1.0 / 0.2),
                1.0 - z,
                (self._torso_rot_z() + 0.7) * (1.0 / 0.7),
            ]
        )

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        constraint_value_before = self.constraint_value()

        self.do_simulation(a, self.frame_skip)
        mjp.functions.mj_rnePostConstraint(self.sim.model, self.sim.data)

        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        constraint_value_after = self.constraint_value()

        yposafter = self.get_body_com("torso")[1]
        obj_cost = float((abs(wall_distances(xposafter, yposafter)) < 1.8).any())
        state = self.state_vector()
        z_rot = self._torso_rot_z()
        upright = state[2] >= 0.2 and state[2] <= 1.0 and z_rot >= -0.7
        violation_cost = float(not upright)
        cost = float(np.clip(obj_cost + violation_cost, 0, 1))

        if self.done_config == "done_on_violation":
            done = not (np.isfinite(state).all() and upright)
        elif self.done_config == "never":
            done = False
        else:
            raise NotImplementedError(f"done_config {self.done_config} is not supported")

        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            cost_obj=obj_cost,
            cost_done=violation_cost,
            cost=cost,
            constraint_value_before=constraint_value_before,
            constraint_value_after=constraint_value_after,
        )

    def _get_obs(self):
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        if x < 20:
            y_off = y - x * _TAN30
        elif 20 < x < 60:
            y_off = y + (x - 40) * _TAN30
        elif 60 < x < 100:
            y_off = y - (x - 80) * _TAN30
        else:
            y_off = y - 20 * _TAN30
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:-42],
                self.sim.data.qvel.flat[:-36],
                [x / 5],
                [y_off],
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qpos[-42:] = self.init_qpos[-42:]
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        qvel[-36:] = self.init_qvel[-36:]
        self.set_state(qpos, qvel)
        return self._get_obs()
