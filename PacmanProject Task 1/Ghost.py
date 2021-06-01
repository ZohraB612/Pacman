from GridObject import GridObject
from Actions import Directions, Actions
import numpy as np
from PacmanEnv import (FOOD, GHOST, AGENT, EMPTY,WALL, REWARD_DIE)


# We define the ghost agent and place it on Pacman's map.

class GhostObs(GridObject):
    def __init__(self, grid, pos, x, y):
        super().__init__(x, y)
        self.grid = grid
        self.ghost_pos = pos

    def ghost_position(self):
        return [self.x, self.y]


""" 
    Ghost object on Pacman's map
"""


class Ghost(GridObject):
    def __init__(self, env, x, y):
        super().__init__(x, y)
        self.direction = Directions.NORTH
        self.steps_left = 5
        self.grid = env.grid
        self.env = env
        self.is_on_food = False

    """ 
        Currently picks a random direction at every step, doing 5 attempts
        Can step on food
    """

    def step(self):
        reward = 0
        num_attempts = 5
        while num_attempts > 0:
            num_attempts -= 1

            direc = np.random.choice(Directions.All)
            next_pos = Actions.get_next_pos(self.get_pos(), direc)
            item = self.grid.get(next_pos)
            if item == WALL or item == GHOST:
                continue

            # if the ghost collides with the agent (when the agent is alive)
            # the agent will die and the reward will be 'REWARD_DIE'
            if item == AGENT and self.env.agent.alive:
                # print("Ghost steps on the pacman ", REWARD_DIE)
                self.env.agent.alive = False
                reward = REWARD_DIE

            self.grid.move_item(self.get_pos(), next_pos, FOOD if self.is_on_food else EMPTY)
            self.set_pos(next_pos[0], next_pos[1])

            self.is_on_food = (item == FOOD)

            break

        return reward
