import os

import torch
import torch.utils.data
from PIL import Image


# TODO: crop box to be within image
# TODO: refactor
# TODO: remove empty boxes


class Dataset(torch.utils.data.Dataset):
    num_classes = 1
    class_names = ['face']

    def __init__(self, path, subset, transform=None):
        self.transform = transform

        if subset == 'train':
            self.images_path = os.path.join(path, 'WIDER_train', 'images')
            self.data = self.load_data(
                os.path.join(os.path.join(path, 'wider_face_split', 'wider_face_train_bbx_gt.txt')))
        elif subset == 'eval':
            self.images_path = os.path.join(path, 'WIDER_val', 'images')
            self.data = self.load_data(
                os.path.join(os.path.join(path, 'wider_face_split', 'wider_face_val_bbx_gt.txt')))
        else:
            raise AssertionError('invalid subset {}'.format(subset))

    def load_data(self, path):
        data = []
        with open(path) as f:
            f = iter(f.read().splitlines())

            try:
                while True:
                    image_path = next(f)
                    num_boxes = int(next(f))

                    if num_boxes == 0:
                        next(f)
                        boxes = []
                    else:
                        boxes = [next(f) for _ in range(num_boxes)]

                    boxes = [[int(x) for x in box.split()[:4]] for box in boxes]
                    boxes = [(t, l, t + h, l + w) for l, t, w, h, in boxes]

                    data.append((image_path, boxes))

            except StopIteration:
                pass

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path, boxes = self.data[item]

        image_path = os.path.join(self.images_path, image_path)
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')

        boxes = torch.tensor(boxes, dtype=torch.float).view(-1, 4)
        class_ids = torch.zeros(boxes.size(0), dtype=torch.long)

        input = {
            'image': image,
            'class_ids': class_ids,
            'boxes': boxes,
        }

        if self.transform is not None:
            input = self.transform(input)

        return input
