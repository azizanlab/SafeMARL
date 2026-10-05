import argparse
import glob
import json
import multiprocessing
import os
import re
import tempfile

import numpy as np
import torch


def run_episode(learner, env_args, seed):
    from safety_ma_mujoco import MujocoMulti

    env = MujocoMulti({**env_args, "eval_mode": True})
    env.env.seed(seed)
    obs, _, _ = env.reset()
    ret, cost = 0.0, 0.0
    for _ in range(env.episode_limit):
        actions = learner.act(np.asarray(obs, dtype=np.float32)[None], stochastic=False)[0]
        obs, _, rewards, dones, infos, _ = env.step(list(actions))
        ret += rewards[0][0]
        cost += infos[0]["cost"][0][0]
        if dones[0]:
            break
    return ret, cost


def step_of(path):
    m = re.search(r"step_(\d+)/madac\.pt$", path)
    return int(m.group(1)) if m else -1


def evaluate(job):
    args, seeds = job
    torch.set_num_threads(1)
    from safe_marl.algorithms.madac import MADAC
    from safety_ma_mujoco import MujocoMulti

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    env_args = dict(cfg["env_args"])
    env = MujocoMulti(env_args)
    learner = MADAC({**cfg["algo_args"]["model"], **cfg["algo_args"]["algo"]}, env.observation_space, env.action_space,
                    env.share_observation_space[0], env.n_agents, torch.device("cpu"))
    results = {}
    for path in sorted(glob.glob(os.path.join(args.run_dir, args.checkpoints)), key=step_of):
        with tempfile.TemporaryDirectory() as tmp:
            os.symlink(os.path.abspath(path), os.path.join(tmp, "madac.pt"))
            learner.restore(tmp)
        results[path] = [run_episode(learner, env_args, seed) for seed in seeds]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoints", default="models/madac.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    seeds = [args.seed + k for k in range(args.episodes)]
    jobs = [(args, seeds[p::args.processes]) for p in range(args.processes)]
    if args.processes > 1:
        with multiprocessing.get_context("spawn").Pool(args.processes) as pool:
            parts = pool.map(evaluate, jobs)
    else:
        parts = [evaluate(jobs[0])]

    print(f"{'checkpoint':>16s} {'return':>8s} {'cost':>7s} {'violating %':>12s} {'>100 steps %':>13s}   ({args.episodes} episodes)")
    for path in sorted(parts[0], key=step_of):
        episodes = [e for part in parts for e in part[path]]
        returns = np.array([e[0] for e in episodes])
        costs = np.array([e[1] for e in episodes])
        name = f"step_{step_of(path)}" if step_of(path) >= 0 else os.path.basename(path)
        print(f"{name:>16s} {returns.mean():8.0f} {costs.mean():7.1f} {100 * (costs > 0).mean():12.1f} "
              f"{100 * (costs > 100).mean():13.1f}", flush=True)


if __name__ == "__main__":
    main()
