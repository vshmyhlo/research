import torch
import torch.nn as nn
import torch.nn.functional as F


class NTM(nn.Module):
    def forward(self):
        pass
        # N = ...
        # M = ...
        #
        # m = torch.zeros(B, N, M, names=('B', 'N', 'M') device=input.device)
        #
        # w = w.softmax(1)
        #
        # r = (w * m).sum(1)
        #
        # # r = memory_read()

        ################
        ################

    pass


def batch_convolve(a, b):
    assert a.size(0) == b.size(0)
    assert a.size(1) == b.size(1)

    output = torch.Tensor(a.size(0), a.size(1), max(a.size(2), b.size(2)) - min(a.size(2), b.size(2)) // 2 * 2)

    for i in range(a.size(0)):
        output[i:i + 1] = F.conv1d(a[i:i + 1], b[i:i + 1])

    return output


def content(m, k, b):
    k = k.unsqueeze(1)
    cos = F.cosine_similarity(k, m, 2)
    w = (b * cos).softmax(1)

    return w


def gate(w_content, w_prev, g):
    w = g * w_content + (1 - g) * w_prev

    return w


def shift(w_gate, s):
    w_gate = w_gate.unsqueeze(1)
    s = s.unsqueeze(1)
    w_gate = F.pad(w_gate, (1, 1), mode='circular')
    w = batch_convolve(w_gate, s)
    w = w.squeeze(1)

    return w


def sharpen(w_shift, y):
    w = w_shift**y
    w = w / w.sum(1, keepdim=True)

    return w


def main():
    # sizes
    B = 10
    N, M = 128, 20

    # memory and input
    m = torch.rand(B, N, M)
    x = torch.rand(B, M + 6)

    # k, b, g, s, y
    read_lengths = [M, 1, 1, 3, 1]
    k, b, g, s, y = torch.split(x, read_lengths, dim=1)

    w_prev = torch.zeros(B, N)

    w_content = content(m, k, b)
    w_gate = gate(w_content, w_prev, g)
    w_shift = shift(w_gate, s)
    w_sharpen = sharpen(w_shift, y)

    w = w_sharpen
    r = memory_read(m, w)

    print(r.shape)


def memory_read(m, w):
    w = w.unsqueeze(2)
    r = (m * w).sum(1)

    return r


def memory_write(m, w, e, a):
    pass


if __name__ == '__main__':
    main()
