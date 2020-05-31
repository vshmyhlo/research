import librosa
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchaudio.functional import istft


class Spectrogram(nn.Module):
    def __init__(self, sample_rate, num_mels, mean_std):
        super().__init__()

        # TODO:
        self.num_fft = round(50. / 1000 * sample_rate)
        self.win_length = round(50. / 1000 * sample_rate)
        self.hop_length = round(12.5 / 1000 * sample_rate)
        self.c = 1.

        filters = torch.tensor(librosa.filters.mel(sample_rate, n_fft=self.num_fft, n_mels=num_mels), dtype=torch.float)
        self.filters = nn.Parameter(filters, requires_grad=False)
        self.filters_inv = nn.Parameter(torch.pinverse(filters), requires_grad=False)

        self.mean, self.std = [
            nn.Parameter(x.view(1, x.size(0), 1), requires_grad=False)
            for x in mean_std]

    def forward(self, input):
        input, _ = self.wave_to_spectra(input)
        return input

    def wave_to_spectra(self, input):
        input = torch.stft(
            input,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=input.device))

        real, imag = torch.unbind(input, dim=-1)
        input = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag, real)

        input = F.conv1d(input, self.filters.unsqueeze(-1))
        # TODO: refactor
        input = self.c * torch.log(torch.clamp(input, min=1e-5))

        # normalize
        input = normalize(input, self.mean, self.std)

        return input, phase

    def spectra_to_wave(self, input, phase):
        # inv normalize
        input = inv_normalize(input, self.mean, self.std)

        # TODO: refactor
        input = torch.exp(input / self.c)
        input = F.conv1d(input, self.filters_inv.unsqueeze(-1))

        input = torch.stack([input * torch.cos(phase), input * torch.sin(phase)], dim=-1)

        input = istft(
            input,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=input.device))

        return input


def normalize(input, mean, std):
    return (input - mean) / std


def inv_normalize(input, mean, std):
    return input * std + mean
