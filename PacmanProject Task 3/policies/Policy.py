import numpy as np
from Actions import Actions
import random
import torch


# Garcia Ortiz, M. (2021). Labs from City, University London. INM707: Deep Reinforcement Learning


class Policy:
    def get_action(self, obs, goal= None):
        pass

    def update(self, obs, action, reward, next_obs, goal):
        pass

    @staticmethod
    def get_random_action():
        return np.random.choice(Actions.All)

    def epsilon_greedy(self):
        eps = 0.5
        p = np.random.random()

        if eps > p:
            return self.get_random_action
        else:
            return GoForFood
