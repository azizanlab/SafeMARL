import copy
from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process

import numpy as np


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    closed = False

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close_extras(self):
        pass

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


def _reset_if_done(env, ob, s_ob, done, info, available_actions):
    if np.all(done):
        info[0]["original_obs"] = copy.deepcopy(ob)
        info[0]["original_state"] = copy.deepcopy(s_ob)
        info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
        ob, s_ob, available_actions = env.reset()
    return ob, s_ob, available_actions


def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            ob, s_ob, available_actions = _reset_if_done(env, ob, s_ob, done, info, available_actions)
            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "reset":
            remote.send(env.reset())
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == "get_num_agents":
            remote.send(env.n_agents)
        else:
            raise NotImplementedError(cmd)


def _costs_from_infos(infos):
    return np.array([info[0]["cost"] for info in infos], dtype=np.float32)


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_num_agents", None))
        self.n_agents = self.remotes[0].recv()
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, nenvs, observation_space, share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(rews),
            _costs_from_infos(infos),
            np.stack(dones),
            infos,
            np.stack(available_actions),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

