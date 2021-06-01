"""
GridObject is a base class for all objects on the grid (Agent, Ghost)
"""
class GridObject:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_pos(self):
        return [self.x, self.y]

    def set_pos(self, x, y):
        self.x = x
        self.y = y
