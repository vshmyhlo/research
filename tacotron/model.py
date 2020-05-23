import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, vocab_size, features=512):
        super().__init__()

        self.encoder = Encoder(vocab_size, features)
        self.decoder = Decoder(features)

    def forward(self, input, target):
        etc = {}

        input = self.encoder(input)
        input = self.decoder(input, target)
       
        return input, etc


class Encoder(nn.Module):
    def __init__(self, vocab_size, features):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, features)
        self.conv = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(features, features, 5, padding=2, bias=False),
                nn.BatchNorm1d(features),
                nn.ReLU(inplace=True))
            for _ in range(3)
        ])
        self.rnn = nn.LSTM(features, features // 2, bidirectional=True, batch_first=True)

    def forward(self, input):
        input = self.emb(input)
        input = input.permute(0, 2, 1)
        input = self.conv(input)
        input = input.permute(0, 2, 1)
        input, _ = self.rnn(input)

        return input


class Decoder(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.pre_net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(features // 2, features // 2),
                nn.BatchNorm1d(features // 2),
                nn.ReLU(inplace=True))
            for _ in range(2)
        ])

        self.output_rnn = nn.LSTM(features * 2, features * 2, bidirectional=False, batch_first=True, num_layers=2)

        self.output_proj = nn.Linear(1, 1)

        self.post_net = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(features, features, 5, padding=2, bias=False),
                nn.BatchNorm1d(features),
                nn.Identity() if i == 4 else nn.Tanh())
            for i in range(5)
        ])

    def forward(self, input, target):
        target = self.pre_net(target)

        for _ in range(target.size(1)):
            pass

        target = torch.cat([target, context], 1)

        target, hidden = self.output_rnn(target)

        target = torch.cat([target, context], 1)

        pre_output = self.output_proj(target)
        target = pre_output + self.post_net(pre_output)

        return target, pre_output


m = Model(10)
x = torch.randint(0, 10, size=(32, 10))

y = m(x)

print(y.shape)
