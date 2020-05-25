import torch
from torch import nn as nn

from tacotron.utils import transpose_t_c


class Decoder(nn.Module):
    def __init__(self, num_mels, features):
        super().__init__()

        self.pre_net = PreNet(num_mels, features)

        self.attention = Attention(features)
        self.rnn_1 = nn.LSTMCell(features + features // 2, features * 2)
        self.rnn_2 = nn.LSTMCell(features * 3, features * 2)

        self.output_proj = nn.Linear(features * 3, num_mels)

    def forward(self, input, input_mask, target):
        target = torch.cat([
            torch.zeros(target.size(0), target.size(1), 1, device=target.device),
            target[:, :, :-1]
        ], 2)
        target = self.pre_net(target)

        state = {
            'context': torch.zeros(input.size(0), input.size(2), device=input.device),
            'attention_weight': torch.zeros(input.size(0), input.size(1), device=input.device),
            'attention_weight_cum': torch.zeros(input.size(0), input.size(1), device=input.device),
            'rnn_1': None,
            'rnn_2': None,
        }

        key = self.attention.key(input)  # FIXME:

        output = []
        weight = []
        for t in range(target.size(2)):
            out, state = self.step(input, input_mask, key, target[:, :, t], state)
            output.append(out)
            weight.append(state['attention_weight'])

        output = torch.stack(output, 2)
        weight = torch.stack(weight, 2)

        return output, weight

    def step(self, input, input_mask, key, output, state):
        state_prime = {}
        context = state['context']

        output = torch.cat([output, context], 1)
        state_prime['rnn_1'] = self.rnn_1(output, state['rnn_1'])
        output, _ = state_prime['rnn_1']

        context, state_prime['attention_weight'] = self.attention(
            query=output,
            key=key,
            value=input,
            mask=input_mask,
            weight=torch.stack([
                state['attention_weight'],
                state['attention_weight_cum'],
            ], 1))
        state_prime['attention_weight_cum'] = state['attention_weight_cum'] + state_prime['attention_weight']

        output = torch.cat([output, context], 1)
        state_prime['rnn_2'] = self.rnn_2(output, state['rnn_2'])
        output, _ = state_prime['rnn_2']

        output = torch.cat([output, context], 1)
        output = self.output_proj(output)

        state_prime['context'] = context

        return output, state_prime


class PreNet(nn.Sequential):
    def __init__(self, num_mels, features):
        blocks = []
        for i in range(2):
            blocks.append(nn.Conv1d(num_mels if i == 0 else features // 2, features // 2, 1, bias=False))
            blocks.append(nn.BatchNorm1d(features // 2))
            blocks.append(nn.ReLU(inplace=True))

        super().__init__(*blocks)


# TODO: init


class Attention(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.query = nn.Linear(features * 2, features // 4, bias=False)
        self.key = nn.Linear(features, features // 4, bias=False)
        self.weight = LocationLayer(features)

        self.scores = nn.Linear(features // 4, 1)

    def forward(self, query, key, value, mask, weight):
        query = self.query(query.unsqueeze(1))
        # key = self.key(key) # FIXME:
        weight = self.weight(weight)

        scores = query + key + weight
        scores = self.scores(torch.tanh(scores))
        scores = scores.masked_fill_(~mask.unsqueeze(-1), float('-inf'))
        weight = scores.softmax(1)

        context = (weight * value).sum(1)

        return context, weight.squeeze(2)


class LocationLayer(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv = nn.Conv1d(2, 32, 31, padding=31 // 2, bias=False)
        self.linear = nn.Linear(32, features // 4, bias=False)

    def forward(self, input):
        input = self.conv(input)
        input = transpose_t_c(input)
        input = self.linear(input)

        return input
