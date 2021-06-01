from policies.RandomPolicy import *
from policies.GoForFood import *
from policies.qLearning import *

"""
Factory method for various policies
"""


def make_policy(name, env, params, verbose=False):
    if name == "random":
        return RandomPolicy()
    if name == "goforfood":
        return GoForFood()
    if name == "qlearning":
        return QLearning(env, params, verbose=verbose)

    return RandomPolicy()
