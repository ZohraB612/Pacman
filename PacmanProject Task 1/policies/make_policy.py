from policies.RandomPolicy import *
from policies.GoForFood import *


"""
Factory method for various policies
"""


def make_policy(name):
    if name == "random":
        return RandomPolicy()
    if name == "goforfood":
        return GoForFood()
    return RandomPolicy()
