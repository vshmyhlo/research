import os

import click
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as T
from all_the_tools.config import load_config
from all_the_tools.torch.metrics import Concat
from all_the_tools.torch.utils import Saver
from tqdm import tqdm

from mela.dataset import FOLDS, Dataset2020Test
from mela.model import Model
from mela.transforms import LoadImage
from transforms import ApplyTo, Extract, Map
from transforms.image import TTA8
from utils import random_seed

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
STD = torch.tensor([0.2470, 0.2435, 0.2616])


@click.command()
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--dataset-path', type=click.Path(), required=True)
@click.option('--experiment-path', type=click.Path(), required=True)
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(config_path, **kwargs):
    config = load_config(
        config_path,
        **kwargs)
    del kwargs
    random_seed(config.seed)

    eval_transform = build_transforms(config)

    eval_dataset = Dataset2020Test(
        os.path.join(config.dataset_path, '2020'), transform=eval_transform)

    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.eval.batch_size // 2,
        shuffle=False,
        drop_last=False,
        num_workers=config.workers)

    all_logits = Concat(1)

    for fold in range(1, FOLDS + 1):
        model = Model(config.model).to(DEVICE)
        saver = Saver({
            'model': model,
        })
        restore_path = os.path.join(config.experiment_path, 'F{}'.format(fold), 'checkpoint_best.pth')
        saver.load(restore_path, keys=['model'])

        logits, all_ids = predict_fold(model, eval_data_loader, fold=fold)

        all_logits.update(logits.cpu())

    all_logits = all_logits.compute()
    print(all_logits.shape)

    all_probs = all_logits.sigmoid().mean(1)
    print(all_probs.shape, len(all_ids))

    submission = pd.DataFrame({
        'image_name': all_ids,
        'target': all_probs.data.cpu().numpy(),
    }).set_index('image_name')

    submission.to_csv(os.path.join(config.experiment_path, 'submission.csv'))


def build_transforms(config):
    eval_transform = T.Compose([
        LoadImage(T.Resize(config.image_size)),
        ApplyTo(
            'image',
            T.Compose([
                T.CenterCrop(config.crop_size),
                TTA8(),
                Map(T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ])),
                T.Lambda(lambda x: torch.stack(x, 0)),
            ])),
        Extract(['image', 'meta', 'id']),
    ])

    return eval_transform


def predict_fold(model, data_loader, fold):
    all_logits = Concat()
    all_ids = []

    with torch.no_grad():
        model.eval()
        for images, meta, ids in tqdm(data_loader, desc='fold {}, infer'.format(fold)):
            images, meta = images.to(DEVICE), {k: meta[k].to(DEVICE) for k in meta}

            b, n, c, h, w = images.size()
            images = images.view(b * n, c, h, w)
            logits = model(images, meta)
            logits = logits.view(b, n)

            all_logits.update(logits.cpu())
            all_ids.extend(ids)

    all_logits = all_logits.compute()

    return all_logits, all_ids


if __name__ == '__main__':
    main()
