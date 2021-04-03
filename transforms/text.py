import numpy as np


class VocabEncode(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, input):
        input = self.vocab.encode(input)
        input = np.array(input, dtype=np.int64)

        return input


class Normalize(object):
    def __call__(self, input):
        input = input.lower()

        return input
