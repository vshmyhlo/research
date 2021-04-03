import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
from tqdm import tqdm


def pad_and_pack(tensors):
    sizes = [t.size(0) for t in tensors]

    tensor = torch.zeros(
        len(sizes),
        max(sizes),
        dtype=tensors[0].dtype,
        layout=tensors[0].layout,
        device=tensors[0].device,
    )
    mask = torch.zeros(
        len(sizes),
        max(sizes),
        dtype=torch.bool,
        layout=tensors[0].layout,
        device=tensors[0].device,
    )

    for i, t in enumerate(tensors):
        tensor[i, : t.size(0)] = t
        mask[i, : t.size(0)] = True

    return tensor, mask


def collate_fn(batch):
    batch = sorted(batch, key=lambda b: b[0].size(0), reverse=True)
    e, d = list(zip(*batch))

    e, e_mask = pad_and_pack(e)
    d, d_mask = pad_and_pack(d)

    return (e, e_mask), (d, d_mask)


def compute_sample_sizes(dataset):
    data = []
    for i in tqdm(range(len(dataset))):
        t, a = dataset[i]
        data.append({"t": t.size(0), "a": a.size(0)})
    data = pd.DataFrame(data)

    data["r"] = data["a"] / data["t"]
    data["size"] = (data["t"] * data["r"].mean() + data["a"]) / 2

    return data["size"]


# TODO:
def griffin_lim(spectra, module, n_iters=30):
    spectra = module.norm(spectra, inverse=True)
    spectra = module.log_mel(spectra, inverse=True)

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*spectra.size())))
    angles = torch.tensor(angles, dtype=torch.float, device=spectra.device)
    audio = module.stft(spectra, angles, inverse=True).squeeze(1)

    for i in range(n_iters):
        _, angles = module.stft(audio)
        audio = module.stft(spectra, angles, inverse=True).squeeze(1)

    return audio


def downsample_mask(input, size):
    assert input.dim() == 2
    assert input.dtype == torch.bool

    input = input.unsqueeze(1).float()
    input = F.interpolate(input, size=size, mode="nearest")
    input = input.squeeze(1).bool()

    return input


def transpose_t_c(input):
    return input.transpose(1, 2)


def soft_diag(h, w):
    def r(x):
        return np.floor(x).astype(np.int64)

    j = np.arange(w)
    i = r(j.astype(np.float) / w * h)
    x = np.zeros((h, w))
    x[i, j] = 1.0

    x = gaussian_filter(x, h / 20, mode="constant")
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 2.0 - 1.0

    return x
