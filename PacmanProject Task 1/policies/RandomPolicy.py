from policies.Policy import *

"""
This policy picks actions at random
"""


class RandomPolicy(Policy):
    def get_action(self, obs):
        return self.get_random_action()
