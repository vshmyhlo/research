import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from fcos.utils import foreground_binary_coding, draw_boxes, Detections
from losses import sigmoid_focal_loss


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1 = (1 - p)**gamma * torch.log(p)
    term2 = p**gamma * torch.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


B = 100
num_classes = 10

logits = torch.empty(B, num_classes).normal_()
target = torch.randint(0, num_classes, size=(B,))

l1 = sigmoid_focal_loss(input=logits, target=foreground_binary_coding(target, num_classes), gamma=2, alpha=0.25)
l2 = sigmoid_focal_loss_cpu(logits=logits, targets=target, gamma=2, alpha=0.25)

assert torch.allclose(l1, l2)

dets = Detections(
    boxes=torch.tensor([
        [128, 128, 128 + 256, 128 + 256],

    ], dtype=torch.float),
    class_ids=torch.tensor([0], dtype=torch.long),
    scores=torch.ones([1], dtype=torch.float))
image = to_tensor(Image.open('./image.jpeg').resize((512, 512)))
image = draw_boxes(image, dets, ['c1', 'c2', 'c3'])
image = to_pil_image(image)
plt.imshow(image)
plt.show()
