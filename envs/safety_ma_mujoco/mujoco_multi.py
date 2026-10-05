import gym
import numpy as np
from gym.spaces import Box
from gym.wrappers import TimeLimit

SCENARIOS = {
    "HalfCheetah-v2": ("2x3", "3x2"),
    "Walker2d-v2": ("2x3", "3x2"),
    "Ant-v2": ("2x4", "4x2"),
}


def parse_agent_conf(agent_conf):
    n_agents, n_act = agent_conf.lower().split("x")
    return [int(n_act)] * int(n_agents)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low, high = self.action_space.low, self.action_space.high
        return low + (action + 1.0) / 2.0 * (high - low)


class MujocoMulti:
    def __init__(self, env_args):
        self.scenario = env_args["scenario"]
        self.agent_conf = env_args["agent_conf"]
        self.episode_limit = int(env_args.get("episode_limit", 1000))
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {self.scenario}; choose from {list(SCENARIOS)}")
        if self.agent_conf not in SCENARIOS[self.scenario]:
            raise ValueError(
                f"agent_conf {self.agent_conf} is not one of {SCENARIOS[self.scenario]} for {self.scenario}"
            )
        self.normalize_obs = self.scenario == "Ant-v2"

        robot_kwargs = {
            "done_config": env_args.get("done_config", "never"),
            "eval_mode": bool(env_args.get("eval_mode", False)),
        }
        if self.scenario == "HalfCheetah-v2":
            from safety_ma_mujoco.half_cheetah import HalfCheetahEnv as robot_cls
        elif self.scenario == "Walker2d-v2":
            from safety_ma_mujoco.walker2d import Walker2dEnv as robot_cls
        else:
            from safety_ma_mujoco.ant import AntEnv as robot_cls

        self.env = robot_cls(**robot_kwargs)
        self.timelimit_env = TimeLimit(self.env, max_episode_steps=self.episode_limit)
        self.wrapped_env = NormalizedActions(self.timelimit_env)

        self.action_dims = parse_agent_conf(self.agent_conf)
        total_act_dim = self.env.action_space.shape[0]
        assert sum(self.action_dims) == total_act_dim, (
            f"agent_conf {self.agent_conf} does not partition the {total_act_dim}-dim action space"
        )
        self.n_agents = len(self.action_dims)
        self.steps = 0

        offsets = np.cumsum([0] + self.action_dims)
        self.action_space = tuple(
            Box(
                self.env.action_space.low[offsets[a]:offsets[a + 1]],
                self.env.action_space.high[offsets[a]:offsets[a + 1]],
                dtype=np.float32,
            )
            for a in range(self.n_agents)
        )

        self.timelimit_env.reset()
        obs_dim = len(self.get_obs()[0])
        bound = 10.0 if self.normalize_obs else np.inf
        self.observation_space = [Box(low=-bound, high=bound, shape=(obs_dim,)) for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=-bound, high=bound, shape=(obs_dim,)) for _ in range(self.n_agents)]

    def step(self, actions):
        flat_actions = np.concatenate(
            [np.asarray(actions[i])[: self.action_dims[i]] for i in range(self.n_agents)]
        )
        _, reward, done, info = self.wrapped_env.step(flat_actions)
        self.steps += 1

        info = dict(info)
        if done:
            info["bad_transition"] = self.steps >= self.episode_limit
        info["cost"] = [[float(info["cost"])]] * self.n_agents
        info["constraint_value_before"] = [[float(info["constraint_value_before"])]] * self.n_agents
        info["constraint_value_after"] = [[float(info["constraint_value_after"])]] * self.n_agents

        rewards = [[reward]] * self.n_agents
        dones = [done] * self.n_agents
        infos = [info] * self.n_agents
        return self.get_obs(), self.get_state(), rewards, dones, infos, self.get_avail_actions()

    def reset(self):
        self.steps = 0
        self.timelimit_env.reset()
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def seed(self, seed):
        self.env.seed(seed)
        self.env.action_space.seed(seed)

    def close(self):
        self.env.close()

    def _agent_vectors(self):
        state = self.env._get_obs()
        vectors = []
        for a in range(self.n_agents):
            agent_id = np.zeros(self.n_agents, dtype=np.float32)
            agent_id[a] = 1.0
            vec = np.concatenate([state, agent_id]).astype(np.float32)
            if self.normalize_obs:
                vec = (vec - vec.mean()) / vec.std()
            vectors.append(vec)
        return vectors

    def get_obs(self):
        return self._agent_vectors()

    def get_state(self):
        return self._agent_vectors()

    def get_avail_actions(self):
        return np.ones((self.n_agents, max(self.action_dims)), dtype=np.float32)
