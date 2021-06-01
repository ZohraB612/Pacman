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
        win = 0


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

            if done and reward > 50000:
                win += 1

        return episode_reward, num_steps, survived, win

    """ 
    runs multiple episodes and aggregates statistics
    """
    def run(self):
        for i in range(self.num_episodes):
            if i % 100 == 0:
                print("Episode #", i)
            rew, steps, surv, win = self.run_episode()
            self.rewards.append(rew)
            self.steps_taken.append(steps)
            self.survived.append(surv)
            self.win_percent.append(win)

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

        smooth = int(0.08 * self.num_episodes)
        x = range(self.num_episodes)
        ax.plot(x, self.moving_average(self.rewards, smooth), color='red', linewidth=1)
        ax.set_ylabel('Rolling Rewards', color = 'red')
        ax1 = ax.twinx()
        ax1.plot(x, self.moving_average(self.steps_taken, smooth), color='blue')
        ax1.set_ylabel('Rolling Steps', color = 'blue')
        ax1.set_xlabel('Number of episodes')
        ax1.set_title('Rolling Rewards and Steps through episodes')
        
        plt.show();
        
        plt.figure(figsize=(20,15))
        
        plt.plot(x, self.moving_average(self.survived, smooth))
        plt.ylabel('Survival rate')
        plt.xlabel('Number of episodes')

        plt.title('Pacman Survived')
        
        plt.show();
        

# testing
if __name__ == "__main__":
    a = np.arange(1, 20)
    print(a)
    print(Game.moving_average(a, 5))
