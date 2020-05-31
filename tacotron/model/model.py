import torch
from torch import nn as nn

from tacotron.model.decoder import Decoder
from tacotron.model.encoder import Encoder
from tacotron.model.modules import ConvNorm1d
from tacotron.model.spectrogram import Spectrogram
from tacotron.utils import downsample_mask


class Model(nn.Module):
    def __init__(self, model, vocab_size, sample_rate, mean_std):
        super().__init__()

        self.spectra = Spectrogram(sample_rate=sample_rate, num_mels=model.num_mels, mean_std=mean_std)
        self.encoder = Encoder(vocab_size=vocab_size, base_features=model.base_features)
        self.decoder = Decoder(num_mels=model.num_mels, base_features=model.base_features)
        self.post_net = PostNet(num_mels=model.num_mels, mid_features=model.base_features)

    def forward(self, input, input_mask, target, target_mask):
        with torch.no_grad():
            target, _ = self.spectra(target)
            target_mask = downsample_mask(target_mask, target.size(2))

        input = self.encoder(input, input_mask)
        pre_output, weight = self.decoder(input, input_mask, target)
        output = pre_output + self.post_net(pre_output)

        return output, pre_output, target, target_mask, weight


class PostNet(nn.Sequential):
    def __init__(self, num_mels, mid_features, num_layers=5):
        blocks = []
        for i in range(num_layers - 1):
            blocks.append(ConvNorm1d(num_mels if i == 0 else mid_features, mid_features, 5, padding=2, init='tanh'))
            blocks.append(nn.Tanh())
            blocks.append(nn.Dropout(0.5))
        blocks.append(ConvNorm1d(mid_features, num_mels, 5, padding=2))
        blocks.append(nn.Dropout(0.5))

        super().__init__(*blocks)
