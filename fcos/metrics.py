import time

from all_the_tools.metrics import Mean, Metric

from detection.map import per_class_precision_recall


class FPS(Mean):
    def __init__(self):
        super().__init__()

        self.t = None

    def update(self, n):
        if self.t is None:
            self.t = time.time()
            return

        t = time.time()
        super().update(n / (t - self.t))
        self.t = t

    def reset(self):
        super().reset()

        self.t = None


# TODO: refactor to do intermediate computation
class PerClassPR(Metric):
    def __init__(self):
        self.true = []
        self.pred = []

    def compute(self):
        return per_class_precision_recall(self.true, self.pred, 0.5)

    def update(self, value):
        true, pred = value
        self.true.extend(true)
        self.pred.extend(pred)

    def reset(self):
        self.true = []
        self.pred = []
