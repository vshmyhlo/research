import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import istft

from tacotron.model.decoder import Decoder
from tacotron.model.encoder import Encoder
from tacotron.utils import downsample_mask


# We transform the STFT magnitude to the mel scale using an 80
# channel mel filterbank spanning 125 Hz to 7.6 kHz, followed by log
# dynamic range compression. Prior to log compression, the filterbank
# output magnitudes are clipped to a minimum value of 0.01 in order
# to limit dynamic range in the logarithmic domain.

class Spectrogram(nn.Module):
    def __init__(self, sample_rate, num_mels):
        super().__init__()

        # TODO:
        self.num_fft = round(50. / 1000 * sample_rate)
        self.win_length = round(50. / 1000 * sample_rate)
        self.hop_length = round(12.5 / 1000 * sample_rate)

        filters = torch.tensor(librosa.filters.mel(sample_rate, n_fft=self.num_fft, n_mels=num_mels), dtype=torch.float)
        self.filters = nn.Parameter(filters, requires_grad=False)
        self.filters_inv = nn.Parameter(torch.pinverse(filters), requires_grad=False)

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
        input = torch.log(torch.clamp(input, min=1e-8))

        return input, phase

    def spectra_to_wave(self, input, phase):
        input = torch.exp(input)
        input = F.conv1d(input, self.filters_inv.unsqueeze(-1))

        input = torch.stack([input * torch.cos(phase), input * torch.sin(phase)], dim=-1)

        input = istft(
            input,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=input.device))

        return input

    def forward(self, input):
        input, _ = self.wave_to_spectra(input)
        return input


class Model(nn.Module):
    def __init__(self, model, vocab_size, sample_rate):
        super().__init__()

        self.spectra = Spectrogram(sample_rate, model.num_mels)
        self.encoder = Encoder(vocab_size, model.base_features)
        self.decoder = Decoder(model.num_mels, model.base_features)
        self.post_net = PostNet(model.num_mels, model.base_features)

        self.apply(self.weights_init)

    def forward(self, input, input_mask, target, target_mask):
        input = self.encoder(input, input_mask)
        target = self.spectra(target)
        target_mask = downsample_mask(target_mask, target.size(2))
        pre_output, weight = self.decoder(input, input_mask, target)
        output = pre_output + self.post_net(pre_output)

        return output, pre_output, target, target_mask, weight

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear,)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Embedding,)):
            nn.init.normal_(m.weight)


class PostNet(nn.Sequential):
    def __init__(self, num_mels, features, num_layers=5):
        blocks = []
        for i in range(num_layers - 1):
            in_features = num_mels if i == 0 else features

            blocks.append(nn.Conv1d(in_features, features, 5, padding=2, bias=False))
            blocks.append(nn.BatchNorm1d(features))
            blocks.append(nn.Tanh())
            blocks.append(nn.Dropout(0.5))

        blocks.append(nn.Conv1d(features, num_mels, 5, padding=2))

        super().__init__(*blocks)
