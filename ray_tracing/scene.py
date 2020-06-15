from typing import List

import torch
from dataclasses import dataclass

from ray_tracing.light import Light
from ray_tracing.objects import Object


@dataclass
class Scene(object):
    camera: torch.Tensor
    objects: List[Object]
    lights: List[Light]
