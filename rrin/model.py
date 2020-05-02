import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rrin.unet import UNet


def weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.)
    elif isinstance(m, (nn.BatchNorm2d,)):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)


def warp(img, flow):
    _, _, H, W = img.size()
    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
    gridX = torch.tensor(gridX, requires_grad=False).cuda()
    gridY = torch.tensor(gridY, requires_grad=False).cuda()
    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    normx = 2 * (x / W - 0.5)
    normy = 2 * (y / H - 0.5)
    grid = torch.stack((normx, normy), dim=3)
    warped = F.grid_sample(img, grid)
    return warped


class Model(nn.Module):
    def __init__(self, level=3):
        super().__init__()

        self.unet_1 = UNet(6, 4, 5)
        self.unet_2 = UNet(10, 4, 4)
        self.unet_3 = UNet(16, 2, 4)
        self.unet_4 = UNet(9, 3, 4)

        self.apply(weight_init)

    def process(self, i_0, i_1, t):
        i = torch.cat((i_0, i_1), 1)

        f_0_1, f_1_0 = self.unet_1(i).split(2, dim=1)

        f = (1 - t) * f_0_1 + t * -f_1_0
        f_hat_t_0 = t * -f
        f_hat_t_1 = (1 - t) * f

        f_tilde_t_0, f_tilde_t_1 = self.unet_2(torch.cat((i, f_hat_t_0, f_hat_t_1), 1)).split(2, dim=1)

        f_t_0 = f_hat_t_0 + f_tilde_t_0
        f_t_1 = f_hat_t_1 + f_tilde_t_1

        i_hat_0_t = warp(i_0, f_t_0)
        i_hat_1_t = warp(i_1, f_t_1)

        m_0, m_1 = F.sigmoid(self.unet_3(torch.cat((i, i_hat_0_t, i_hat_1_t, f_t_0, f_t_1), 1))).split(1, dim=1)

        w_0, w_1 = (1 - t) * m_0, t * m_1
        i_hat_t_c = (w_0 * i_hat_0_t + w_1 * i_hat_1_t) / (w_0 + w_1 + 1e-8)

        etc = {
            'flow_t_0': f_t_0,
            'flow_t_1': f_t_1,
            'mask_0': m_0,
            'mask_1': m_1,
        }

        return i_hat_t_c, etc

    def forward(self, i_0, i_1, t=0.5):
        i_hat_t_c, etc = self.process(i_0, i_1, t)
        i_tilde_t = self.unet_4(torch.cat((i_0, i_1, i_hat_t_c), 1))
        i_hat_t = i_hat_t_c + i_tilde_t
        # final = final  # .clamp(0,1)
      
        return i_hat_t, etc
