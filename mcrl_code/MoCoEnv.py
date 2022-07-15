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


class MoCoCartpoleEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.env = gym.make('CartPole-v0')
        self.last_reward = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        obs = self.env.reset()
        self.last_reward = 0
        return obs

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        self.last_reward += reward
        if done:
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
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        obs = self.env.reset()
        self.last_reward = 0
        return obs

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        # self.last_reward += reward
        self.last_reward += (new_obs[1] * new_obs[1] * 10)
        self.last_reward += reward
        if done:
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
            return new_obs, self.total_reward * 1000 + self.total_step + 2000, done, info
        else:
            return new_obs, 0, done, info

    def seed(self, seed=None):
        np.random.seed(seed)
