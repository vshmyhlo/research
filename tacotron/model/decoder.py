import torch
from torch import nn as nn

from tacotron.model.attention import Attention
from tacotron.model.modules import ConvNorm1d, Linear


class Decoder(nn.Module):
    def __init__(self, num_mels, base_features):
        super().__init__()

        self.pre_net = PreNet(num_mels=num_mels, out_features=base_features // 2)
        self.attention = Attention(
            query_features=base_features * 2, key_features=base_features, mid_features=base_features // 4)
        self.rnn_attention = nn.LSTMCell(base_features + base_features // 2, base_features * 2)
        self.rnn_decoder = nn.LSTMCell(base_features + base_features * 2, base_features * 2)
        self.rnn_dropout = nn.Dropout(0.1)
        self.output_proj = Linear(base_features + base_features * 2, num_mels)

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
        output = self.rnn_dropout(output)

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
        state_prime['rnn_decoder'] = self.rnn_decoder(output, state['rnn_decoder'])
        output, _ = state_prime['rnn_decoder']
        output = self.rnn_dropout(output)

        output = torch.cat([output, context], 1)
        output = self.output_proj(output)

        state_prime['context'] = context

        return output, state_prime


class PreNet(nn.Sequential):
    def __init__(self, num_mels, out_features):
        blocks = []
        for i in range(2):
            blocks.append(ConvNorm1d(num_mels if i == 0 else out_features, out_features, 1, init='relu'))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(0.5))

        super().__init__(*blocks)
