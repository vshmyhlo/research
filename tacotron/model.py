import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import istft

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

        # self.mel = nn.Conv1d(filters.shape[1], filters.shape[0], 1, bias=False)
        # self.mel.weight.data.copy_(self.filters_to_tensor(filters))
        # self.mel.weight.requires_grad = False

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

    def forward(self, input, input_mask, target, target_mask):
        input = self.encoder(input)
        target = self.spectra(target)
        target_mask = downsample_mask(target_mask, target.size(2))
        output, pre_output, weight = self.decoder(input, input_mask, target)

        return output, pre_output, target, target_mask, weight


class Encoder(nn.Module):
    def __init__(self, vocab_size, features):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, features)
        self.conv = EncoderConv(features)
        self.rnn = nn.LSTM(features, features // 2, bidirectional=True, batch_first=True)

    def forward(self, input):
        input = self.emb(input)
        input = swap_t_c(input)
        input = self.conv(input)
        input = swap_t_c(input)
        input, _ = self.rnn(input)

        return input


class Decoder(nn.Module):
    def __init__(self, num_mels, features):
        super().__init__()

        self.pre_net = PreNet(num_mels, features)

        self.attention = Attention(features)
        self.rnn_1 = nn.LSTMCell(features + features // 2, features * 2)
        self.rnn_2 = nn.LSTMCell(features * 3, features * 2)

        self.output_proj = nn.Linear(features * 3, num_mels)
        self.post_net = PostNet(num_mels, features)

    def forward(self, input, input_mask, target):
        target = self.pre_net(target)
        target = torch.cat([
            torch.zeros(target.size(0), target.size(1), 1, device=target.device),
            target[:, :, :-1]
        ], 2)

        state = {
            'context': torch.zeros(input.size(0), input.size(2), device=input.device),
            'attention_weight': torch.zeros(input.size(0), input.size(1), device=input.device),
            'rnn_1': None,
            'rnn_2': None,
        }

        pre_output = []
        weight = []
        for t in range(target.size(2)):
            out, state = self.step(input, input_mask, target[:, :, t], state)
            pre_output.append(out)
            weight.append(state['attention_weight'])

        pre_output = torch.stack(pre_output, 2)
        weight = torch.stack(weight, 2)

        output = pre_output + self.post_net(pre_output)

        return output, pre_output, weight

    def step(self, input, input_mask, output, state):
        state_prime = {}
        context = state['context']

        output = torch.cat([output, context], 1)
        state_prime['rnn_1'] = self.rnn_1(output, state['rnn_1'])
        output, _ = state_prime['rnn_1']

        context, state_prime['attention_weight'] = self.attention(
            query=output, key_value=input, mask=input_mask, weight=state['attention_weight'])

        output = torch.cat([output, context], 1)
        state_prime['rnn_2'] = self.rnn_2(output, state['rnn_2'])
        output, _ = state_prime['rnn_2']

        output = torch.cat([output, context], 1)
        output = self.output_proj(output)

        state_prime['context'] = context

        return output, state_prime


class Attention(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.weight_conv = nn.Conv1d(1, 32, 31, padding=31 // 2, bias=False)

        self.query = nn.Linear(features * 2, features // 4, bias=False)
        self.key = nn.Linear(features, features // 4, bias=False)
        self.weight = nn.Linear(32, features // 4, bias=False)
        self.bias = nn.Parameter(torch.Tensor(1, features // 4))

        self.scores = nn.Linear(features // 4, 1, bias=False)

        nn.init.zeros_(self.bias)

    def forward(self, query, key_value, mask, weight):
        key = value = key_value
        del key_value

        query = self.query(query.unsqueeze(1))
        key = self.key(key)
        weight = self.weight_conv(weight.unsqueeze(1))
        weight = self.weight(swap_t_c(weight))

        scores = query + key + weight + self.bias
        scores = self.scores(torch.tanh(scores))
        scores = scores.masked_fill_(~mask.unsqueeze(-1), float('-inf'))
        weight = scores.softmax(1)

        context = (weight * value).sum(1)

        return context, weight.squeeze(2)


class EncoderConv(nn.Sequential):
    def __init__(self, features):
        blocks = []
        for _ in range(3):
            blocks.append(nn.Conv1d(features, features, 5, padding=2, bias=False))
            blocks.append(nn.BatchNorm1d(features))
            blocks.append(nn.ReLU(inplace=True))

        super().__init__(*blocks)


class PreNet(nn.Sequential):
    def __init__(self, num_mels, features):
        blocks = []
        blocks.append(nn.BatchNorm1d(num_mels))
        for i in range(2):
            blocks.append(nn.Conv1d(num_mels if i == 0 else features // 2, features // 2, 1, bias=False))
            blocks.append(nn.BatchNorm1d(features // 2))
            blocks.append(nn.ReLU(inplace=True))

        super().__init__(*blocks)


class PostNet(nn.Sequential):
    def __init__(self, num_mels, features):
        blocks = []
        for i in range(5):
            in_features = num_mels if i == 0 else features
            out_features = num_mels if i == 4 else features

            blocks.append(nn.Conv1d(in_features, out_features, 5, padding=2, bias=False))
            if i == 4:
                continue
            blocks.append(nn.BatchNorm1d(features))
            blocks.append(nn.Tanh())

        super().__init__(*blocks)


def swap_t_c(input):
    return input.permute(0, 2, 1)
