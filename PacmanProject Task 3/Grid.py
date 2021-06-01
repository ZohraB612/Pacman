"""
 The grid maintains the map and objects on the map as
 a list of lists of characters.
"""


class Grid:

    def __init__(self, grid):
        self.grid = [list(row) for row in grid]
        self.n_cols = len(grid[0])
        self.n_rows = len(grid)

        for row in self.grid:
            if len(row) != self.n_cols:
                raise Exception("Grid must rectangular")

    def reset(self):
        pass

    """
    get item from the given position (r,c) (where r represents
    rows and c represents columns)
    """
    
    def get(self, pos):
        return self.grid[pos[0]][pos[1]]

    """
    set an item on a given position (r,c)
    """
    
    def set(self, pos, data):
        self.grid[pos[0]][pos[1]] = data

    def is_valid(self, i, j):
        return 0 <= i < self.n_rows and 0 <= j < self.n_cols

    def move_item(self, pos, next_pos, new_item):
        item = self.get(pos)
        self.set(next_pos, item)
        self.set(pos, new_item)

    def display(self):
        for row in self.grid:
            print("".join(row))
