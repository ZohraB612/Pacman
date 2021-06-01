import pandas as pd
from Maps import *
from policies.make_policy import *
from PacmanEnv import PacmanEnv
from PacmanEnv import REWARD_DIE
import matplotlib.pyplot as plt

"""
    Learning system implementation
    Runs episodes, interacting between agent and the environment
"""


class Game:

    def __init__(self, policy="random",
                 num_episodes=1,
                 num_steps_per_episode=5,
                 num_ghosts=1,
                 mapp=map20,
                 verbose=False):
        self.verbose = verbose
        self.env = PacmanEnv(num_ghosts=num_ghosts, mapp=mapp)
        self.num_episodes = num_episodes
        self.max_steps_per_episode = num_steps_per_episode
        self.policy = make_policy(policy)

        # statistics for episodes
        self.rewards = []
        self.steps_taken = []
        self.survived = []
        self.win_percent = []

    """
    runs a single episode
    """

    def run_episode(self):
        obs = self.env.reset()
        if self.verbose:
            self.env.grid.display()
        episode_reward = 0

        done = False

        num_steps = 0
        survived = 1

        while not done and num_steps < self.max_steps_per_episode:

            num_steps += 1
            action = self.policy.get_action(obs)

            if self.verbose:
                print(f"Step:{num_steps}, Action:{action}")

            reward, next_obs, done = self.env.step(action)
            self.policy.update(obs, action, reward, next_obs)
            episode_reward += reward
            if self.verbose:
                print("Reward:", reward, " Episode Reward: ", episode_reward)
                self.env.grid.display()
            obs = next_obs

            if done and reward < REWARD_DIE:
                survived = 0

        return episode_reward, num_steps, survived

    """ 
    runs multiple episodes and aggregates statistics
    """

    def run(self):
        for i in range(self.num_episodes):
            if i % 100 == 0:
                print("Episode #", i)
            rew, steps, surv = self.run_episode()
            self.rewards.append(rew)
            self.steps_taken.append(steps)
            self.survived.append(surv)

