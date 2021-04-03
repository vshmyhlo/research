import cv2
import numpy as np

#
# flow = np.array([
#     [1, 0],
#     [-1, 0],
#     [1, -0.01],
#     [0, 0],
# ], dtype=np.float32)
#
# print(flow)
#
# flow = flow.reshape((2, 2, 2))
#
# mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#
# mag = mag.reshape((4,))
# ang = ang.reshape((4,))
#
# # print(ang)
# # print(ang * 180 / np.pi / 2)
# # print()
# # print(mag)
# # print(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX))
#
#
rgb = np.array(
    [
        [255, 0, 0],
        [255, 255, 255],
        [0, 0, 0],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)

hsv = cv2.cvtColor(rgb.reshape((2, 2, 3)), cv2.COLOR_RGB2HSV).reshape((4, 3))
print(hsv)

import numpy as np
import torch
from PIL import Image
from rrin2.train import flow_to_rgb

y, x = torch.meshgrid(torch.linspace(-1, 1, 128), torch.linspace(-1, 1, 128))

flow = torch.stack([x, y], 0)
image = flow_to_rgb(flow.unsqueeze(0)).squeeze(0)

image = image.permute(1, 2, 0).data.cpu().numpy()
image = (image * 255).astype(np.uint8)

print(image.shape, image.dtype, image.min(), image.max())
Image.fromarray(image).save("flow.png")
