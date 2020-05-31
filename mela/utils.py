import torch
from all_the_tools.metrics import Metric


class Concat(Metric):
    def reset(self):
        self.values = []

    def update(self, value):
        self.values.append(value)

    def compute(self):
        return torch.cat(self.values, 0)
