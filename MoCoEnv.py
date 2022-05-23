"""
# @Author: JuQi
# @Time  : 2022/2/7 4:10 PM
# @E-mail: 18672750887@163.com
"""
from __future__ import print_function
import gym
from gym.spaces import Discrete, Box
import numpy as np
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class MoCoTSPEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.last_reward = 0
        self.total_point_num = 10
        self.now_point = 0
        self.start_point = 0
        self.now_step = 0
        self.action_space = Box(0.0, 2 * np.pi, shape=(1,), dtype=np.float32)
        self.observation_space = Box(0.0, 1.0, shape=(self.total_point_num, 2,), dtype=np.float32)
        # self.xy = np.array([
        #     [0.5, 0.5],
        #     [1, 0.5],
        #     [1, 1],
        #     [0.5, 1]
        # ])
        np.random.seed(0)
        self.xy = np.random.rand(self.total_point_num, 2)
        self.history = []

        self.obs = np.zeros((self.total_point_num, 2))

    def reset(self):
        self.obs = np.zeros((self.total_point_num, 2))
        self.obs[0, 0] = 1
        self.obs[0, 1] = 1
        self.now_point = 0
        self.last_reward = 0
        self.now_step = 0
        self.history = []
        return self.obs

    def step(self, action):
        self.history.append(action)
        self.now_step += 1
        tmp_xy = self.xy - self.xy[self.now_point, :]
        angle = np.arctan2(tmp_xy[:, 1], tmp_xy[:, 0])
        ans = (angle - action)
        ans = ans * ans
        ans = ans + self.obs[:, 1] * 100
        ans[self.now_point] += 100
        next_point = np.argmin(ans)
        self.obs[:, 0] = 0
        self.obs[next_point, 0] = 1
        self.obs[next_point, 1] = 1

        self.last_reward = self.last_reward - self.get_len(next_point, self.now_point)
        self.now_point = next_point
        if self.now_step == self.total_point_num - 1:
            new_obs = np.zeros((self.total_point_num, 2))
            new_obs[:, 1] = 1
            if self.last_reward - self.get_len(self.start_point, self.now_point) > -1.8:
                assert False, self.history
            return new_obs, self.last_reward - self.get_len(self.start_point, self.now_point), True, {}
        else:
            return self.obs, 0, False, {}

    def get_len(self, a, b):
        tmp_x = self.xy[a, 0] - self.xy[b, 0]
        tmp_y = self.xy[a, 1] - self.xy[b, 1]
        return np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)

    def seed(self, seed=None):
        np.random.seed(seed)


class MoCoCartpoleEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.env = gym.make('CartPole-v0')
        self.last_reward = 0
        self.total_step = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        obs = self.env.reset()
        self.last_reward = 0
        self.total_step = 0
        return obs

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        self.last_reward += reward
        self.total_step += 1
        if done:
            info['len'] = self.total_step
            return new_obs, self.last_reward, done, info
        else:
            return new_obs, 0, done, info

    def seed(self, seed=None):
        np.random.seed(seed)


class MoCoMountainCarEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.env = gym.make('MountainCar-v0')
        self.last_reward = 0
        self.total_step = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        obs = self.env.reset()
        self.last_reward = 0
        self.total_step = 0
        return obs

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        # self.last_reward += reward
        self.last_reward += (new_obs[1] * new_obs[1] * 10)
        self.last_reward += reward
        self.total_step += 1
        if done:
            info['len'] = self.total_step
            return new_obs, self.last_reward + new_obs[0], done, info
        else:
            return new_obs, 0, done, info

    def seed(self, seed=None):
        np.random.seed(seed)


class MoCoPongEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.env = gym.make('Pong-ramDeterministic-v4')
        self.last_reward = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.total_step = 0
        self.total_reward = 0

    def reset(self):
        self.total_step = 0
        self.total_reward = 0
        obs = self.env.reset()
        return obs

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        self.total_step += 1
        self.total_reward += reward
        if self.total_step > 499 or self.total_reward == -2:
            done = True
        if done:
            info['len'] = self.total_step
            return new_obs, self.total_reward + self.total_step * 0.001, done, info
        else:
            return new_obs, 0, done, info

    def seed(self, seed=None):
        np.random.seed(seed)


class MoCoPongSplitEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.env = gym.make('Pong-ramDeterministic-v4')
        self.last_reward = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.total_step = 0
        self.total_reward = 0
        self.change_time = 0
        self.is_real_done = True
        self.obs = self.env.reset()

    def reset(self):
        self.total_reward = 0
        self.change_time = 0
        self.total_step = 0
        if self.is_real_done:
            self.obs = self.env.reset()
            self.is_real_done = False
        return self.obs

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        self.total_step += 1
        self.obs = new_obs
        if done:
            self.is_real_done = True
        self.total_reward += reward
        if reward != 0:
            self.change_time += 1
            if self.change_time == 1:
                done = True
        if done:
            if self.total_reward > 0:
                return self.obs, self.total_reward + 0.001 * self.total_step, done, info
            else:
                return self.obs, self.total_reward + 0.001 * self.total_step, done, info
        else:
            return self.obs, 0, done, info


class MoCoPongSplitVideoEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.env = gym.make('PongDeterministic-v4')
        self.last_reward = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.total_step = 0
        self.total_reward = 0
        self.change_time = 0
        self.is_real_done = True
        self.obs = self.env.reset()

    def reset(self):
        self.total_reward = 0
        self.change_time = 0
        self.total_step = 0
        if self.is_real_done:
            self.obs = self.env.reset()
            self.is_real_done = False
        return self.obs

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        self.total_step += 1
        self.obs = new_obs
        if done:
            self.is_real_done = True
        self.total_reward += reward
        if reward != 0:
            self.change_time += 1
            if self.change_time == 1:
                done = True
        if done:
            if self.total_reward > 0:
                return self.obs, self.total_reward + 0.001 * self.total_step, done, info
            else:
                return self.obs, self.total_reward + 0.001 * self.total_step, done, info
        else:
            return self.obs, 0, done, info


class MoCoAlienEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.env = gym.make('Alien-ramDeterministic-v4')
        self.last_reward = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.total_step = 0
        self.total_reward = 0

    def reset(self):
        self.total_step = 0
        self.total_reward = 0
        obs = self.env.reset()
        return obs

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        self.total_step += 1
        self.total_reward += reward
        # if self.total_step > 499 or self.total_reward == -2:
        #     done = True
        if done:
            return new_obs, self.total_reward, done, info
        else:
            return new_obs, 0, done, info

    def seed(self, seed=None):
        np.random.seed(seed)


class MoCoBreakoutEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.env = gym.make('Breakout-ramDeterministic-v4')
        self.last_reward = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.total_step = 0
        self.total_reward = 0

    def reset(self):
        self.total_step = 0
        self.total_reward = 0
        obs = self.env.reset()
        return obs

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        self.total_step += 1
        self.total_reward += reward
        if self.total_step == 999:
            done = True
        if done:
            return new_obs, self.total_reward + self.total_step * 0.001, done, info
        else:
            return new_obs, 0, done, info


class MoCoFreewayEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.env = gym.make('Freeway-ramDeterministic-v4')
        self.last_reward = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.total_step = 0
        self.total_reward = 0

    def reset(self):
        self.total_step = 0
        self.total_reward = 0
        obs = self.env.reset()
        return obs

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        self.total_step += 1
        self.total_reward += reward
        if self.total_step == 499:
            done = True
        if done:
            return new_obs, self.total_reward, done, info
        else:
            return new_obs, 0, done, info


class MoCoSpaceInvadersEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.env = gym.make('SpaceInvadersDeterministic-v4')
        self.last_reward = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.total_step = 0
        self.total_reward = 0

    def reset(self):
        # 不更新
        self.total_step = 0
        self.total_reward = 0
        obs = self.env.reset()
        return obs

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        self.total_step += 1
        self.total_reward += reward
        return new_obs, reward, done, info
