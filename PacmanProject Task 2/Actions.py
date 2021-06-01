# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# We start by defining the possible movement directions.
# There are 4 Directions: North, South, East, and West.


"""
Possible movement directions
"""


class Directions:
    NORTH = 'North'
    SOUTH = 'South'
    EAST = 'East'
    WEST = 'West'
    All = [NORTH, SOUTH, EAST, WEST]


# Next, we define the agent's possible actions. Currently, the actions are
# identical to the Directions, but they will be defined more in depth later.

"""
Possible agent actions
Currently idential to directions but may e extended later
"""


class Actions:
    NORTH = Directions.NORTH
    SOUTH = Directions.SOUTH
    EAST = Directions.EAST
    WEST = Directions.WEST

    All = [NORTH, SOUTH, EAST, WEST]
    action2id = {a: i for i, a in enumerate(All)}
    # id2action = {i: a for i, a in enumerate(All)}

    REVERSE = {NORTH: SOUTH,
               SOUTH: NORTH,
               EAST: WEST,
               WEST: EAST}

    @staticmethod
    def id2action(i):
        return Actions.All[i]

    # The following code represents a collection of static methods that
    #allow us to manipulate the move actions.
    

    """
    A collection of static methods for manipulating move actions.
    """

    # Directions - helps to find next position after movement
    directions = {Directions.NORTH: (-1, 0),
                  Directions.SOUTH: (1, 0),
                  Directions.EAST: (0, 1),
                  Directions.WEST: (0, -1)}

    @staticmethod
    def get_next_pos(pos, action):
        move = Actions.directions[action]
        return pos[0] + move[0], pos[1] + move[1]
