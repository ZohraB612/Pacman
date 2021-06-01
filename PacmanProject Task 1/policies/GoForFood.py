from PacmanEnv import (FOOD)
from Actions import Directions
from Actions import Actions
from policies.Policy import Policy


# policy for the agent to maximise food collection
class GoForFood(Policy):
    def get_action(self, obs):
        for dir in Directions.All:
            next_pos = Actions.get_next_pos(obs.agent_pos, dir)
            item = obs.grid.get(next_pos)

            if item == FOOD:
                return dir

        return self.get_random_action()
