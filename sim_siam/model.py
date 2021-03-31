import torch.nn as nn
from torchvision.models import resnet34, resnet50


class Model(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        if backbone == "resnet34":
            self.backbone = resnet34(pretrained=True)
            features = 512
        elif backbone == "resnet50":
            self.backbone = resnet50(pretrained=True)
            features = 2048
        else:
            raise ValueError("invalid backbone {}".format(backbone))

        self.backbone.fc = nn.Identity()
        self.proj = Projector(features)
        self.pred = Predictor(features)

    def forward(self, input):
        input = self.backbone(input)
        z = self.proj(input)
        p = self.pred(z)

        return p, z


class Projector(nn.Sequential):
    def __init__(self, features):
        super().__init__(
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            #
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            #
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
        )


class Predictor(nn.Sequential):
    def __init__(self, features):
        super().__init__(
            nn.Linear(features, features // 4),
            nn.BatchNorm1d(features // 4),
            nn.ReLU(inplace=True),
            #
            nn.Linear(features // 4, features),
        )
