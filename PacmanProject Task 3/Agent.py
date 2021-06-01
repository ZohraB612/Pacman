from GridObject import GridObject
import numpy as np
from Actions import Actions
from Ghost import GhostObs

# Then we define our Pacman Agent class

FOOD = "."
GHOST = "g"

# We define the Agent observation: grid and agent position.

"""
Agent Observation: grid and agent position
"""


class AgentObs:
    # STATE FEATURES
    # 1) agent_pos N*N
    # 2) Which Quadrant has more food : SE, SW, NE, NW:  4
    # 3) Which quadrant has a ghost
    # Quadrant is K x K square

    def __init__(self, grid, pos):
        self.grid = grid
        self.agent_pos = pos

    @staticmethod
    def get_max_state_nums(env):
        return env.grid.n_rows * env.grid.n_cols * 4 * 4

    """
    check if a ghost is on an adjacent square
    """
    
    def is_ghost_there(self, dir):
        next_pos = Actions.get_next_pos(self.agent_pos, dir)
        return self.grid.get(next_pos) == GHOST

    def print_state(self, state_id ):
        quad = {0: "NW", 1: "NE", 2: "SW", 3: "SE"}

        ghost_dir = state_id % 4
        state_id = state_id // 4

        food_dir = state_id % 4
        state_id = state_id // 4

        pos_col = state_id % self.grid.n_cols
        state_id = state_id // self.grid.n_cols

        pos_row = state_id

        print(f"State ID:{state_id} row:{pos_row}, col:{pos_col} food_dir:{quad[food_dir]} ghost_dir:{quad[ghost_dir]}")

    """
    gets state vector ( x, y, food_dir, ghost_dir) 
    could be extended later with additional features
    """
    def get_state_vector(self):
        # We start by setting the size of the quadrants
        K = 3
        corners = [(-K, -K), # NW
                   (-K, 1),  # NE
                   (1, -K),  # SW
                   (1, 1)]   # SE

        food_counts = []
        ghost_counts = []
        for c in corners:
            start_x = self.agent_pos[0] + c[0]
            start_y = self.agent_pos[1] + c[1]
            food_count = 0
            ghost_count = 0
            
            # We get the food_count and ghost_count within the 4 quadrants
            # around the agent's position
            for x in range(start_x, start_x + K + 1):
                for y in range(start_y, start_y + K + 1):
                    if not self.grid.is_valid(x, y):
                        continue
                    food_count += self.grid.get([x, y]) == FOOD
                    ghost_count += self.grid.get([x, y]) == GHOST

            # We update the quadrant statistics
            food_counts.append(food_count)
            ghost_counts.append(ghost_count)

        # We set for the agent to find the quadrants with the most food
        # and the most ghosts
        food_dir = np.argmax(food_counts)
        ghost_dir = np.argmax(ghost_counts)

        return np.array([self.agent_pos[0], self.agent_pos[1] , food_dir, ghost_dir])

    """ 
    maps the current state vector to a single state_id 
    """
    def get_state(self):
        vec = self.get_state_vector(self)

        return vec[0] * (self.grid.n_cols * 4 * 4) + \
               vec[1] * (4 * 4) + vec[2] * 4 + vec[3]

"""
 GridObject representing the pacman, used in PacmanEnv
"""

class Agent(GridObject):

    def __init__(self, x, y):
        super().__init__(x, y)
        self.speed = 2
        self.alive = True
