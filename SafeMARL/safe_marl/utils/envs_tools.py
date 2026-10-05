import os
import random

import numpy as np
import torch
from safety_ma_mujoco import MujocoMulti, ShareSubprocVecEnv


def check(value):
    return torch.from_numpy(value) if isinstance(value, np.ndarray) else value


def get_shape_from_obs_space(obs_space):
    assert obs_space.__class__.__name__ == "Box", "only Box observation spaces are supported"
    return obs_space.shape


def get_shape_from_act_space(act_space):
    assert act_space.__class__.__name__ == "Box", "only continuous action spaces are supported"
    return act_space.shape[0]


def _env_args_for(env_args, eval_mode):
    return {
        "scenario": env_args["scenario"],
        "agent_conf": env_args["agent_conf"],
        "episode_limit": env_args.get("episode_limit", 1000),
        "done_config": env_args.get("done_config", "never"),
        "eval_mode": eval_mode,
    }


def _make_vec_env(env_name, n_threads, env_args, eval_mode, seed_fn):
    if env_name != "mujoco":
        raise NotImplementedError(f"environment {env_name} is not supported; only 'mujoco' is")

    def get_env_fn(rank):
        def init_env():
            env = MujocoMulti(_env_args_for(env_args, eval_mode))
            env.seed(seed_fn(rank))
            return env

        return init_env

    return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_train_env(env_name, seed, n_threads, env_args):
    return _make_vec_env(env_name, n_threads, env_args, False, lambda rank: seed + rank * 1000)


def make_eval_env(env_name, seed, n_threads, env_args):
    return _make_vec_env(env_name, n_threads, env_args, True, lambda rank: seed * 50000 + rank * 10000)


def set_seed(args):
    if args["seed"] is None:
        args["seed"] = int(np.random.randint(0, 10000))
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def get_num_agents(env, env_args, envs):
    return envs.n_agents
