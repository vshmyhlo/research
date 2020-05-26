import torch
from torch import nn as nn

from tacotron.utils import transpose_t_c


class Decoder(nn.Module):
    def __init__(self, num_mels, base_features):
        super().__init__()

        self.pre_net = PreNet(num_mels=num_mels, out_features=base_features // 2)
        self.attention = Attention(
            query_features=base_features * 2, key_features=base_features, mid_features=base_features // 4)
        self.rnn_attention = nn.LSTMCell(base_features + base_features // 2, base_features * 2)
        self.rnn_decoder = nn.LSTMCell(base_features + base_features * 2, base_features * 2)
        self.output_proj = nn.Linear(base_features + base_features * 2, num_mels)
        self.dropout = nn.Dropout(0.5)

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
            'rnn_attention': None,
            'rnn_decoder': None,
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
        state_prime['rnn_attention'] = self.rnn_attention(output, state['rnn_attention'])
        output, _ = state_prime['rnn_attention']

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

        output = self.dropout(output)
        output = torch.cat([output, context], 1)
        state_prime['rnn_decoder'] = self.rnn_decoder(output, state['rnn_decoder'])
        output, _ = state_prime['rnn_decoder']

        output = torch.cat([output, context], 1)
        output = self.output_proj(output)

        state_prime['context'] = context

        return output, state_prime


class PreNet(nn.Sequential):
    def __init__(self, num_mels, out_features):
        blocks = []
        for i in range(2):
            blocks.append(nn.Conv1d(num_mels if i == 0 else out_features, out_features, 1, bias=False))
            blocks.append(nn.BatchNorm1d(out_features))
            blocks.append(nn.ReLU(inplace=True))

        super().__init__(*blocks)


class Attention(nn.Module):
    def __init__(self, query_features, key_features, mid_features):
        super().__init__()

        self.query = nn.Linear(query_features, mid_features, bias=False)
        self.key = nn.Linear(key_features, mid_features, bias=False)
        self.weight = LocationLayer(mid_features)

        self.scores = nn.Linear(mid_features, 1)

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
        self.linear = nn.Linear(32, features, bias=False)

    def forward(self, input):
        input = self.conv(input)
        input = transpose_t_c(input)
        input = self.linear(input)

        return input
