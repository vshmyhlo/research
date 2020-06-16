from ray_tracing.ray import Ray
from ray_tracing.vector import vector


class Camera(object):
    def __init__(self, origin):
        self.origin = origin

    def ray_to_position(self, x, y):
        position = vector(2 * x - 1, 2 * y - 1, 0)

        return Ray(self.origin, position - self.origin)
