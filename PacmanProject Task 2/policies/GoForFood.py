from PacmanEnv import (FOOD)
from Actions import Directions
from Actions import Actions
from policies.Policy import Policy

# CREATE DIFFERENT AGENT POLICIES
# 1) Random
# 2) Go for the food

# return a direction of movement


class GoForFood(Policy):
    def get_action(self, obs):
        for dir in Directions.All:
            next_pos = Actions.get_next_pos(obs.agent_pos, dir)
            item = obs.grid.get(next_pos)

            if item == FOOD:
                return dir

        return self.get_random_action()

# TODO:
# 1) not step on a ghost
# 2) don't go in the direct of ghost ( 2 steps )
# 3) q value/ dyno q ?
