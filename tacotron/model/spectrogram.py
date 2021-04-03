import librosa
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchaudio.functional import istft

from tacotron.model.modules import Invertible


class Spectrogram(Invertible):
    def __init__(self, sample_rate, num_mels, mean_std):
        super().__init__()

        num_fft = round(50.0 / 1000 * sample_rate)
        win_length = round(50.0 / 1000 * sample_rate)
        hop_length = round(12.5 / 1000 * sample_rate)

        self.stft = STFT(num_fft, win_length, hop_length)
        self.log_mel = LogMel(sample_rate, num_fft, num_mels)
        self.norm = Normalization(mean_std)

    def f(self, input):
        input, phase = self.stft(input)
        input = self.log_mel(input)
        input = self.norm(input)

        return input, phase

    def inv_f(self, input, phase):
        input = self.norm(input, inverse=True)
        input = self.log_mel(input, inverse=True)
        input = self.stft(input, phase, inverse=True)

        return input


class STFT(Invertible):
    def __init__(self, num_fft, win_length, hop_length):
        super().__init__()

        self.num_fft = num_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def f(self, input):
        input = torch.stft(
            input,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=input.device),
        )

        real, imag = torch.unbind(input, dim=-1)
        input = torch.sqrt(real ** 2 + imag ** 2)
        phase = torch.atan2(imag, real)

        return input, phase

    def inv_f(self, input, phase):
        input = torch.stack([input * torch.cos(phase), input * torch.sin(phase)], dim=-1)

        input = istft(
            input,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=input.device),
        )

        return input


class LogMel(Invertible):
    def __init__(self, sample_rate, num_fft, num_mels):
        super().__init__()

        filters = torch.tensor(
            librosa.filters.mel(sample_rate, n_fft=num_fft, n_mels=num_mels), dtype=torch.float
        )
        self.filters = nn.Parameter(filters, requires_grad=False)
        self.inv_filters = nn.Parameter(torch.pinverse(filters), requires_grad=False)
        self.c = 1.0

    def f(self, input):
        input = F.conv1d(input, self.filters.unsqueeze(-1))
        input = torch.clamp(input, min=1e-5)
        input = self.c * torch.log(input)

        return input

    def inv_f(self, input):
        input = torch.exp(input / self.c)
        input = F.conv1d(input, self.inv_filters.unsqueeze(-1))

        return input


class Normalization(Invertible):
    def __init__(self, mean_std):
        super().__init__()

        self.mean, self.std = [
            nn.Parameter(x.view(1, x.size(0), 1), requires_grad=False) for x in mean_std
        ]

    def f(self, input):
        input = (input - self.mean) / self.std

        return input

    def inv_f(self, input):
        input = input * self.std + self.mean

        return input
