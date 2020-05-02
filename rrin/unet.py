import torch
from torch import nn


# Adapted from "Tunable U-Net implementation in PyTorch"
# https://github.com/jvanvugt/pytorch-unet


class ReLU(nn.LeakyReLU):
    def __init__(self, ):
        super().__init__(0.1, inplace=True)


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=2,
            depth=5,
            wf=5):
        super().__init__()

        self.relu = ReLU()
        self.pool = nn.AvgPool2d(2, 2)

        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2**(wf + i)))
            prev_channels = 2**(wf + i)

        self.midconv = nn.Conv2d(prev_channels, prev_channels, kernel_size=3, padding=1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2**(wf + i)))
            prev_channels = 2**(wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=3, padding=1)

    def forward(self, input):
        blocks = []
        for i, layer in enumerate(self.down_path):
            input = layer(input)
            if i == len(self.down_path) - 1:
                continue
            blocks.append(input)
            input = self.pool(input)

        input = self.midconv(input)
        input = self.relu(input)

        for i, layer in enumerate(self.up_path):
            input = layer(input, blocks[-i - 1])

        input = self.last(input)

        return input


class UNetConvBlock(nn.Sequential):
    def __init__(self, in_size, out_size):
        super().__init__(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            ReLU(),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            ReLU())


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1))
        self.conv_block = UNetConvBlock(in_size, out_size)

    # def center_crop(self, layer, target_size):
    #     _, _, layer_height, layer_width = layer.size()
    #     diff_y = (layer_height - target_size[0]) // 2
    #     diff_x = (layer_width - target_size[1]) // 2
    #     return layer[
    #            :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
    #            ]

    def forward(self, input, bridge):
        input = self.up(input)
        # bridge = self.center_crop(bridge, input.shape[2:])
        output = torch.cat((input, bridge), 1)
        output = self.conv_block(output)

        return output
