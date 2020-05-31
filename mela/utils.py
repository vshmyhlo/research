import torch
from all_the_tools.metrics import Metric


class Concat(Metric):
    def __init__(self, dim=None):
        super().__init__()

        self.dim = dim
       
    def reset(self):
        self.values = []

    def update(self, value):
        self.values.append(value)

    def compute(self):
        return torch.cat(self.values, dim=self.dim)
