import math

import torch
import torch.nn as nn

from tacotron.model.modules import ConvNorm1d
from tacotron.utils import transpose_t_c


class Encoder(nn.Module):
    def __init__(self, vocab_size, base_features):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, base_features, padding_idx=0)
        self.conv = EncoderConv(features=base_features)
        self.rnn = nn.LSTM(base_features, base_features // 2, bidirectional=True, batch_first=True)

        std = math.sqrt(2.0 / (vocab_size + base_features))
        val = math.sqrt(3.0) * std
        nn.init.uniform_(self.emb.weight, -val, val)

    def forward(self, input, input_mask):
        input = self.emb(input)
        input = transpose_t_c(input)
        input = self.conv(input)
        input = transpose_t_c(input)

        input = torch.nn.utils.rnn.pack_padded_sequence(input, input_mask.sum(1), batch_first=True)
        input, _ = self.rnn(input)
        input, _ = torch.nn.utils.rnn.pad_packed_sequence(input, batch_first=True, padding_value=0.)

        return input


class EncoderConv(nn.Sequential):
    def __init__(self, features):
        blocks = []
        for _ in range(3):
            blocks.append(ConvNorm1d(features, features, 5, padding=2, init='relu'))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(0.5))

        super().__init__(*blocks)
