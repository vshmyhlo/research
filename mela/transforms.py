import os

import pydicom
from PIL import Image


class LoadImage(object):
    def __init__(self, transform):
        self.transform = transform
        self.cache_path = './cache-mela-images/{}'.format(self.transform)

        os.makedirs(self.cache_path, exist_ok=True)

    def __call__(self, input):
        cache_path = os.path.join(self.cache_path, '{}.png'.format(input['id']))

        if not os.path.exists(cache_path):
            dicom = pydicom.dcmread(input['image'])
            image = Image.fromarray(dicom.pixel_array)
            image = self.transform(image)
            image.save(cache_path)
            del image

        image = Image.open(cache_path)

        input = {
            **input,
            'image': image,
        }

        return input
