from ray_tracing.vector import normalize


class Ray(object):
    def __init__(self, orig, direction):
        self.origin = orig
        self.direction = normalize(direction)

    def position_at(self, t: float):
        return self.origin + t * self.direction
