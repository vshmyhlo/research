import click
import pandas as pd
import torch
import torchvision.transforms as T
from tqdm import tqdm

from detection.box_utils import boxes_aspect_ratio, boxes_area
from detection.config import build_default_config
from detection.datasets.coco import Dataset as CocoDataset
from detection.datasets.wider import Dataset as WiderDataset
from detection.transform import Resize


@click.command()
@click.option('--config-path', type=str, required=True)
@click.option('--dataset-path', type=str, required=True)
def main(config_path, dataset_path):
    config = build_default_config()
    config.merge_from_file(config_path)
    config.freeze()

    if config.dataset == 'coco':
        Dataset = CocoDataset
    elif config.dataset == 'wider':
        Dataset = WiderDataset
    else:
        raise AssertionError('invalid config.dataset {}'.format(config.dataset))

    train_transform = T.Compose([
        Resize(config.resize_size),
    ])
    train_dataset = Dataset(dataset_path, subset='train', transform=train_transform)

    image_boxes = []
    boxes = []
    for input in tqdm(train_dataset):
        image_boxes.append((0, 0, input['image'].size[1], input['image'].size[0]))
        boxes.append(input['boxes'])
    image_boxes = torch.tensor(image_boxes, dtype=torch.float)
    boxes = torch.cat(boxes, 0).float()

    image_stats = pd.DataFrame({
        'size': boxes_area(image_boxes).sqrt().data.cpu().numpy(),
        'aspect_ratio': boxes_aspect_ratio(image_boxes).data.cpu().numpy(),
    })
    box_stats = pd.DataFrame({
        'size': boxes_area(boxes).sqrt().data.cpu().numpy(),
        'aspect_ratio': boxes_aspect_ratio(boxes).data.cpu().numpy(),
    })

    print(image_stats.describe())
    print(box_stats.describe())


if __name__ == '__main__':
    main()
