import os
import pickle
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

from summary_writers.utils import convert_image, convert_scalar, sanitize_tag


class SummaryWriter(object):
    def __init__(self, path):
        self.path = path

    def add_scalar(self, tag, scalar, global_step, close=True):
        tag = sanitize_tag(tag)
        scalar = convert_scalar(scalar)

        with self.load_cache() as cache:
            if tag not in cache["scalars"]:
                cache["scalars"][tag] = []
            cache["scalars"][tag] = [(s, v) for s, v in cache["scalars"][tag] if s < global_step]
            cache["scalars"][tag].append((global_step, scalar))
            steps, values = zip(*cache["scalars"][tag])
            xticks = np.array(steps)
            xticks = (
                np.linspace(xticks.min(), xticks.max(), min(10, len(xticks)))
                .round()
                .astype(np.int32)
            )
            assert len(xticks) <= 10

        fig = plt.figure()
        plt.plot(steps, values)
        plt.xlabel("global_step")
        plt.ylabel(tag)
        plt.xticks(xticks)
        fig.savefig(
            os.path.join(self.build_global_step_dir(global_step), "scalar_{}.jpg".format(tag))
        )
        if close:
            plt.close(fig)

    def add_image(self, tag, image, global_step):
        tag = sanitize_tag(tag)
        image = convert_image(image)
        image.save(
            os.path.join(self.build_global_step_dir(global_step), "image_{}.jpg".format(tag))
        )

    def add_figure(self, tag, fig, global_step, close=True):
        tag = sanitize_tag(tag)
        fig.savefig(
            os.path.join(self.build_global_step_dir(global_step), "figure_{}.jpg".format(tag))
        )
        if close:
            plt.close(fig)

    def flush(self):
        pass

    def close(self):
        pass

    def build_global_step_dir(self, global_step):
        path = os.path.join(self.path, "epoch_{:08d}".format(global_step))
        os.makedirs(path, exist_ok=True)
        return path

    @contextmanager
    def load_cache(self):
        os.makedirs(self.path, exist_ok=True)
        cache_path = os.path.join(self.path, "cache.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        else:
            cache = {"scalars": {}}

        try:
            yield cache
        finally:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
