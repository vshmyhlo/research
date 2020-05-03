import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ReLU(nn.ReLU):
    pass


class Norm(nn.BatchNorm2d):
    pass


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            Norm(out_channels))


class ConvTransposeNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            Norm(out_channels))


class UpsampleMerge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bottom = nn.Sequential(
            ConvNorm(in_channels, out_channels, 3, padding=1),
            ReLU(inplace=True))

        self.refine = nn.Sequential(
            ConvNorm(out_channels, out_channels, 3, padding=1),
            ReLU(inplace=True))

    def forward(self, bottom, left):
        bottom = self.bottom(bottom)
        bottom = F.interpolate(bottom, scale_factor=2, mode='bilinear')
        bottom = crop_to(bottom, left.size()[2:])
        input = self.refine(bottom + left)

        return input


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained=True)

    def forward(self, input):
        input = self.model.conv1(input)
        input = self.model.bn1(input)
        input = self.model.relu(input)
        c1 = input
        input = self.model.maxpool(input)
        input = self.model.layer1(input)
        c2 = input
        input = self.model.layer2(input)
        c3 = input
        input = self.model.layer3(input)
        c4 = input
        input = self.model.layer4(input)
        c5 = input

        return [None, c1, c2, c3, c4, c5]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # self.merge1 = UpsampleMerge(256, 64)
        self.merge2 = UpsampleMerge(128, 64)
        self.merge3 = UpsampleMerge(256, 128)
        self.merge4 = UpsampleMerge(512, 256)

    def forward(self, fmaps):
        input = fmaps[5]

        input = self.merge4(input, fmaps[4])
        input = self.merge3(input, fmaps[3])
        input = self.merge2(input, fmaps[2])
        # input = self.merge1(input, fmaps[1])

        return input


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output = nn.Conv2d(64, num_classes, 1)

        self.decoder.apply(weight_init)
        self.output.apply(weight_init)
       
    def forward(self, input):
        _, _, h, w = input.size()

        input = self.encoder(input)
        input = self.decoder(input)
        input = self.output(input)
        input = F.interpolate(input, scale_factor=4, mode='bilinear')
        input = crop_to(input, (h, w))

        return input


def crop_to(input, size):
    h, w = size

    return input[:, :, :h, :w]


def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm2d,)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
