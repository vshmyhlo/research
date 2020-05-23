import librosa
import numpy as np


class LoadAudio(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, input):
        input, rate = librosa.core.load(input, sr=None, dtype=np.float32)
        assert rate == self.sample_rate

        return input
