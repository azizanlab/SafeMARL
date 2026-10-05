import json
import os
import sys
import time

import numpy as np
import setproctitle
import torch
from pathlib import Path

from macpo.config import get_config
from safety_ma_mujoco import MujocoMulti, ShareSubprocVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env_args = {"scenario": all_args.scenario, "agent_conf": all_args.agent_conf, "episode_limit": 1000,
                        "done_config": all_args.done_config}
            env = MujocoMulti(env_args=env_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env_args = {"scenario": all_args.scenario, "agent_conf": all_args.agent_conf, "episode_limit": 1000,
                        "eval_mode": True}
            env = MujocoMulti(env_args=env_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario", type=str, default="HalfCheetah-v2")
    parser.add_argument("--agent_conf", type=str, default="2x3")
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    if all_args.seed is None:
        all_args.seed = int(np.random.randint(0, 10000))
    if all_args.algorithm_name != "macpo":
        raise NotImplementedError

    if torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name \
        / all_args.scenario / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    run_dir = run_dir / ("seed-%05d-%s" % (all_args.seed, time.strftime("%Y-%m-%d-%H-%M-%S")))
    os.makedirs(str(run_dir))
    with open(str(run_dir / "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(all_args), f, indent=2, sort_keys=True)

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name))

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = envs.n_agents

    config = {"all_args": all_args, "envs": envs, "eval_envs": eval_envs, "num_agents": num_agents, "device": device,
              "run_dir": run_dir}

    from macpo.runner.separated.mujoco_runner_macpo import MujocoRunner as Runner

    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
