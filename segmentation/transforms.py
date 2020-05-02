import random

import numpy as np
import torch
import torchvision.transforms.functional as F


class ToTensor(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, input):
        image = F.to_tensor(input['image'])
        mask = torch.from_numpy(np.array(input['mask'], dtype=np.int64, copy=False))
        mask = torch.eye(self.num_classes)[mask]
        mask = mask.permute(2, 0, 1)

        input = {
            **input,
            'image': image,
            'mask': mask
        }

        return input

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            input = {
                **input,
                'image': F.hflip(input['image']),
                'mask': F.hflip(input['mask'])
            }
           
        return input

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

# class Normalize(object):
#     def __init__(self, mean, std, inplace=False):
#         self.mean = mean
#         self.std = std
#         self.inplace = inplace
#
#     def __call__(self, input):
#         image, mask = input
#
#         image = F.normalize(image, self.mean, self.std, self.inplace)
#
#         return image, mask
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
#
#
# class Resize(object):
#     def __init__(self, size, interpolation=Image.BILINEAR):
#         assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
#         self.size = size
#         self.interpolation = interpolation
#
#     def __call__(self, input):
#         image, mask = input
#         assert image.size == mask.size
#
#         image = F.resize(image, self.size, self.interpolation)
#         mask = F.resize(mask, self.size, Image.NEAREST)
#
#         return image, mask
#
#     def __repr__(self):
#         interpolate_str = T._pil_interpolation_to_str[self.interpolation]
#         return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)
#
#
# class RandomCrop(object):
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#
#     @staticmethod
#     def get_params(img, output_size):
#         w, h = img.size
#         th, tw = output_size
#
#         if w == tw and h == th:
#             return 0, 0, h, w
#
#         i = random.randint(0, h - th)
#         j = random.randint(0, w - tw)
#
#         return i, j, th, tw
#
#     def __call__(self, input):
#         image, mask = input
#         assert image.size == mask.size
#
#         i, j, h, w = self.get_params(image, self.size)
#
#         image = F.crop(image, i, j, h, w)
#         mask = F.crop(mask, i, j, h, w)
#
#         return image, mask
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(size={0})'.format(self.size)
#
#
# class CenterCrop(object):
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#
#     def __call__(self, input):
#         image, mask = input
#         assert image.size == mask.size
#
#         image = F.center_crop(image, self.size)
#         mask = F.center_crop(mask, self.size)
#
#         return image, mask
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(size={0})'.format(self.size)
