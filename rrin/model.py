import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rrin.unet import UNet


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


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.flow = UNet(6, 4, 5)
#         self.refine_flow = UNet(10, 4, 4)
#         self.mask = UNet(16, 2, 4)
#         self.final = UNet(9, 3, 4)
#
#     def process(self, input_0, input_1, t):
#         input_0_1 = torch.cat((input_0, input_1), 1)
#         flow = self.flow(input_0_1)
#
#         Flow_0_1, Flow_1_0 = flow[:, :2, :, :], flow[:, 2:4, :, :]
#         flow_t_0 = -(1 - t) * t * Flow_0_1 + t * t * Flow_1_0
#         flow_t_1 = (1 - t) * (1 - t) * Flow_0_1 - t * (1 - t) * Flow_1_0
#         Flow_t = torch.cat((flow_t_0, flow_t_1, input_0_1), 1)
#         Flow_t = self.refine_flow(Flow_t)
#         flow_t_0 = flow_t_0 + Flow_t[:, :2, :, :]
#         flow_t_1 = flow_t_1 + Flow_t[:, 2:4, :, :]
#
#         input_0_t = warp(input_0, flow_t_0)
#         input_1_t = warp(input_1, flow_t_1)
#
#         temp = torch.cat((flow_t_0, flow_t_1, input_0_1, input_0_t, input_1_t), 1)
#         Mask = F.sigmoid(self.mask(temp))
#         w1, w2 = (1 - t) * Mask[:, 0:1, :, :], t * Mask[:, 1:2, :, :]
#         output = (w1 * input_0_t + w2 * input_1_t) / (w1 + w2 + 1e-8)
#
#         return output
#
#     def forward(self, input_0, input_1, t=0.5):
#         output = self.process(input_0, input_1, t)
#
#         final = torch.cat((input_0, input_1, output), 1)
#         final = self.final(final)
#         final = final + output
#         final = final.clamp(0, 1)
#
#         return final

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.unet_1 = UNet(6, 4, 5)
        self.unet_2 = UNet(10, 4, 4)
        self.unet_3 = UNet(16, 2, 4)
        self.unet_4 = UNet(9, 3, 4)

    def process(self, input_0, input_1, t):
        flow = self.unet_1(torch.cat((input_0, input_1), 1))
        flow_0_1, flow_1_0 = flow[:, :2, :, :], flow[:, 2:, :, :]

        flow_hat_0_1 = (1 - t) * flow_0_1 + t * -flow_1_0

        flow_hat_t_0 = t * -flow_hat_0_1
        flow_hat_t_1 = (1 - t) * flow_hat_0_1

        flow_tilde = self.unet_2(torch.cat((input_0, input_1, flow_hat_t_0, flow_hat_t_1), 1))
        flow_tilde_t_0 = flow_tilde[:, :2, :, :]
        flow_tilde_t_1 = flow_tilde[:, 2:, :, :]

        flow_t_0 = flow_hat_t_0 + flow_tilde_t_0
        flow_t_1 = flow_hat_t_1 + flow_tilde_t_1

        input_hat_0_t = warp(input_0, flow_t_0)
        input_hat_1_t = warp(input_1, flow_t_1)

        m = self.unet_3(torch.cat((input_0, input_1, input_hat_0_t, input_hat_1_t, flow_t_0, flow_t_1), 1))
        m_0 = m[:, 0:1, :, :]
        m_1 = m[:, 1:2, :, :]
        m_t = torch.sigmoid(m_0) / torch.sigmoid(m_1)

        a_t = (t * m_t) / (1 - t)
        b_t = (1 - t) / (t * m_t)
        input_hat_t_c = 1 / (1 + a_t) * input_hat_0_t + 1 / (1 + b_t) * input_hat_1_t

        etc = {
            'flow_hat_0_1': flow_hat_0_1,

            'flow_hat_t_0': flow_hat_t_0,
            'flow_hat_t_1': flow_hat_t_1,

            'flow_tilde_t_0': flow_tilde_t_0,
            'flow_tilde_t_1': flow_tilde_t_1,

            'flow_t_0': flow_t_0,
            'flow_t_1': flow_t_1,
        }

        return input_hat_t_c, etc

    def forward(self, input_0, input_1, t=0.5):
        input_hat_t_c, etc = self.process(input_0, input_1, t)

        input_tilde_t = self.unet_4(torch.cat((input_0, input_1, input_hat_t_c), 1))
        input_hat_t = input_hat_t_c + input_tilde_t

        return input_hat_t, etc
