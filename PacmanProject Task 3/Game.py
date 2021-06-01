# import numpy as np
import pandas as pd
from Maps import *
from policies.make_policy import *
from PacmanEnv import PacmanEnv
from HER import DQNAgentWithHER
import matplotlib.pyplot as plt
import seaborn as sns

# Garcia Ortiz, M. (2021). Labs from City, University London. INM707: Deep Reinforcement Learning

"""
    Learning system implementation
    Runs episodes, interacting between agent and the environment
"""



class Game:
    """
    The Game manages the control flow, soliciting actions from agents.
    """

    def __init__(self, policy="random",
                 params=None,
                 num_episodes=1,
                 num_steps_per_episode=5,
                 num_ghosts=1,
                 mapp=small_map,
                 verbose=False):
        self.verbose = verbose
        self.env = PacmanEnv(num_ghosts=num_ghosts, mapp=mapp)
        self.num_episodes = num_episodes
        self.max_steps_per_episode = num_steps_per_episode
        self.policy = make_policy(policy, self.env, params, verbose)
        # statistics for episodes
        self.rewards = []
        self.steps_taken = []
        self.survived = []
        self.eps_hist = []

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
        transitions = []

        goal_state = obs.get_state_vector()
        while not done and num_steps < self.max_steps_per_episode:
            # print("=" * 30)

            num_steps += 1
            action = self.policy.get_action(obs, goal_state)
            if self.verbose:
                print(f"Step:{num_steps}, Action:{action}")

            reward, next_obs, done = self.env.step(action)
            state_vec = obs.get_state_vector()
            next_state_vec = next_obs.get_state_vector()

            # specific for HER agent(policy) only
            if type(self.policy) is DQNAgentWithHER.DQNAgentWithHER:
                self.policy.update(state_vec, action, reward, next_state_vec, done, goal_state)
                transitions.append((state_vec, action, reward, next_state_vec))
                self.policy.learn()
                if not done and num_steps % 32 == 0:
                    goal_state = self.make_new_goal(state_vec, goal_state, transitions)
            else:
                self.policy.update(obs, action, reward, next_obs, done, goal_state)

            episode_reward += reward
            if self.verbose:
                print("Reward:", reward, " Episode Reward: ", episode_reward)
                self.env.grid.display()
            obs = next_obs

            if done and reward < -1000:
                survived = 0

        return episode_reward, num_steps, survived

    """
    calculates a new goal for HER policy
    """

    def make_new_goal(self, state, goal, transitions):
        new_goal = np.copy(state)
        if np.array_equal(new_goal, goal):
            return new_goal

        for p in range(0, min(32, len(transitions))):
            transition = transitions[p]
            if np.array_equal(transition[3], new_goal):
                self.policy.update(transition[0], transition[1], transition[2],
                                       transition[3], True, new_goal)
                self.policy.learn()
                break

            self.policy.update(transition[0], transition[1], transition[2],
                                   transition[3], False, new_goal)
            self.policy.learn()

        return new_goal

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

            self.eps_hist.append(self.policy.epsilon)

        print('Epsilon History:', self.eps_hist)

    @staticmethod
    def moving_average(x, window):
        s = pd.Series(x)
        res = s.rolling(window).mean()
        return res.values

    """ 
    reports and plots learning progress through the history
    """

    def report_results(self):

        plt.figure(figsize=(30, 25))
        fig, ax = plt.subplots()

        smooth = int(0.2 * self.num_episodes)
        x = range(self.num_episodes)
        sns.lineplot(x, self.moving_average(self.rewards, smooth), color='red', ax=ax)
        ax.set_ylabel('Rolling Rewards', color='red')
        ax1 = plt.twinx()
        sns.lineplot(x, self.moving_average(self.steps_taken, smooth), color='blue', ax = ax1)
        ax1.set_ylabel('Rolling Steps', color='blue')
        ax1.set_xlabel('Number of episodes')
        plt.title('Rolling Rewards and Steps through episodes')

        plt.show();

        plt.figure(figsize=(20, 15))

        plt.plot(x, self.moving_average(self.survived, smooth))
        plt.ylabel('Survival rate')
        plt.xlabel('Number of episodes')

        plt.title('Pacman Survived')

        plt.show();

        plt.figure(figsize=(20, 15))
        
        plt.plot(x, self.moving_average(self.eps_hist, smooth))
        plt.ylabel('Epsilon')
        plt.xlabel('Number of episodes')
        
        plt.title('Epsilon history')
        
        plt.show();


# testing
if __name__ == "__main__":
    a = np.arange(1, 20)
    print(a)
    print(Game.moving_average(a, 5))
