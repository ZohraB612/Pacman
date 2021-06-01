import numpy as np
from Grid import Grid
from Ghost import Ghost
from Agent import AgentObs, Agent
# from policies.AgentSAC import Agent
from Actions import Actions
from Maps import *

# grid characters
EMPTY = " "
FOOD = "."
GHOST = "g"
AGENT = "a"
WALL = "x"

# reward structure
REWARD_WALL = -300
REWARD_FOOD = 10
REWARD_FINISH_FOOD = 10000
REWARD_DIE = -10000
REWARD_TIME = -1

"""
PacmanEnv that simulates the pacman game on a fixed grid,
ghost and allows a pacman (agent) to move around and eat food
"""


class PacmanEnv:

    def __init__(self, mapp=small_map, num_ghosts=2):

        self.num_ghosts = num_ghosts
        self.map = mapp
        self.grid = Grid(mapp)
        self.ghosts = []
        self.agent = None
        self.foodLeft = 0
        self.verbose = False

    """
    The reset function randomly places the ghosts and the agent
    """

    def reset(self):
        # rest the grid
        self.grid = Grid(self.map)

        # First we place ghosts
        num_placed = 0
        self.ghosts = []
        while num_placed < self.num_ghosts:
            # r represents rows
            # c represents columns
            r = np.random.randint(0, self.grid.n_rows)
            c = np.random.randint(0, self.grid.n_cols)

            if self.grid.get([r, c]) == FOOD:
                self.grid.set([r, c], GHOST)
                self.ghosts.append(Ghost(self, r, c))
                num_placed += 1

        # Second we place the agent
        self.agent = None
        while self.agent is None:
            r = np.random.randint(0, self.grid.n_rows)
            c = np.random.randint(0, self.grid.n_cols)
            if self.grid.get([r, c]) == FOOD:
                self.grid.set([r, c], AGENT)
                self.agent = Agent(r, c)

        # And third we count the food
        self.foodLeft = 0
        for r in range(self.grid.n_rows):
            for c in range(self.grid.n_cols):
                if self.grid.get([r, c]) == FOOD:
                    self.foodLeft += 1

        return self.make_agent_obs()

    """
    The move_agent function performs an agent movement
    and calculates reward accordingly.
    """

    def move_agent(self, action):
        # print(action)
        reward = 0
        next_pos = Actions.get_next_pos(pos=self.agent.get_pos(), action=action)
        item = self.grid.get(next_pos)
        # If the agent hits a wall it receives a negative reward: REWARD_WALL
        if item == WALL:
            return REWARD_WALL

        # if the agent steps on food, it receives a positive reward: REWARD_FOOD
        # After the agent steps on food we deduct it from the foodLeft count
        if item == FOOD:
            reward = REWARD_FOOD
            self.foodLeft -= 1

        # if the agent steps on a ghost, it receives a negative reward: REWARD_DIE
        # and it dies
        elif item == GHOST:
            if self.verbose:
                print("Stepped on a ghost ", REWARD_DIE)
            reward = REWARD_DIE
            self.agent.alive = False
        else:
            pass

        if item == FOOD or item == EMPTY:
            self.grid.move_item(self.agent.get_pos(), next_pos, EMPTY)
            self.agent.set_pos(next_pos[0], next_pos[1])

        # print("move agent reward: ", reward)
        return reward

    """
    The move_ghosts function moves all the ghosts.
    """

    def move_ghosts(self):
        reward = 0
        for ghost in self.ghosts:
            reward += ghost.step()
        return reward

    """
    The make_agent_obs creates an agent observation.
    """

    def make_agent_obs(self):
        return AgentObs(self.grid, pos=self.agent.get_pos())

    """
    The is_done function returns that the game is over
    if all food is collected or PacMan agent is dead.
    """

    def is_done(self):
        return self.foodLeft == 0 or not self.agent.alive

    """
    The step function is to be called by the Game.
    """

    def step(self, action):
        reward = self.move_agent(action)
        reward += self.move_ghosts()

        obs = self.make_agent_obs()
        is_done = self.is_done()

        reward += REWARD_TIME
        if self.foodLeft == 0:
            reward += REWARD_FINISH_FOOD

        return reward, obs, is_done

    """
    visual display of the environment
    """

    def display(self):
        print("Food left ", self.foodLeft)
        self.grid.display()
