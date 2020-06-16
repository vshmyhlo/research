from typing import List

from ray_tracing.light import Light
from ray_tracing.objects import Object


class Scene(object):
    def __init__(self, camera, objects: List[Object], lights: List[Light]):
        self.camera = camera
        self.objects = objects
        self.lights = lights
