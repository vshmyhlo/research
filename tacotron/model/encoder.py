import torch
import torch.nn as nn

from tacotron.utils import transpose_t_c


class Encoder(nn.Module):
    def __init__(self, vocab_size, features):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, features)
        self.conv = EncoderConv(features)
        self.rnn = nn.LSTM(features, features // 2, bidirectional=True, batch_first=True)

    def forward(self, input, input_mask):
        input = self.emb(input)
        input = transpose_t_c(input)
        input = self.conv(input)
        input = transpose_t_c(input)
      
        input = torch.nn.utils.rnn.pack_padded_sequence(input, input_mask.sum(1), batch_first=True)
        input, _ = self.rnn(input)
        input, tmp = torch.nn.utils.rnn.pad_packed_sequence(input, batch_first=True, padding_value=0.)

        return input


class EncoderConv(nn.Sequential):
    def __init__(self, features):
        blocks = []
        for _ in range(3):
            blocks.append(nn.Conv1d(features, features, 5, padding=2, bias=False))
            blocks.append(nn.BatchNorm1d(features))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(0.5))

        super().__init__(*blocks)
