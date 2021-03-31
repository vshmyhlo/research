from torch import nn as nn

from cycle_gan.model.modules import LeakyReLU, NoiseBroadcast


class Gen(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            NoiseBroadcast(),
            LeakyReLU(),
        )
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1),
        )

        self.apply(self.init)

    def init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input):
        input = self.input(input)
        input = self.encoder(input)
        input = self.decoder(input)
        input = self.output(input)

        return input


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = EncoderBlock(16, 32)
        self.c2 = EncoderBlock(32, 64)
        self.c3 = EncoderBlock(64, 128)
        self.c4 = EncoderBlock(128, 256)
        self.c5 = EncoderBlock(256, 512)

    def forward(self, input):
        c1 = input = self.c1(input)
        c2 = input = self.c2(input)
        c3 = input = self.c3(input)
        c4 = input = self.c4(input)
        c5 = input = self.c5(input)

        return [None, c1, c2, c3, c4, c5]


class EncoderBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=0.5),
            nn.BatchNorm2d(out_channels),
            NoiseBroadcast(),
            LeakyReLU(),
        )


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.p4 = DecoderMergeBlock(512, 256)
        self.p3 = DecoderMergeBlock(256, 128)
        self.p2 = DecoderMergeBlock(128, 64)
        self.p1 = DecoderMergeBlock(64, 32)
        self.p0 = DecoderBlock(32, 16)

    def forward(self, fmaps):
        input = fmaps[5]
        input = self.p4(input, fmaps[4])
        input = self.p3(input, fmaps[3])
        input = self.p2(input, fmaps[2])
        input = self.p1(input, fmaps[1])
        input = self.p0(input)

        return input


class DecoderMergeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.input = DecoderBlock(in_channels, out_channels)
        self.merge = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            NoiseBroadcast(),
            LeakyReLU(),
        )

    def forward(self, input, lateral):
        input = self.input(input)
        input = input + lateral
        input = self.merge(input)

        return input


class DecoderBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            NoiseBroadcast(),
            LeakyReLU(),
        )
