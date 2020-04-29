import os
from itertools import islice

import click
import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
from all_the_tools.config import load_config
from tensorboardX import SummaryWriter
from ticpfptp.metrics import Mean
from tqdm import tqdm

import utils
from rrin.dataset import Dataset, ForEach, RandomTemporalFlip, RandomHorizontalFlip
from rrin.model import Net as Model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--dataset-path', type=click.Path(), required=True)
@click.option('--experiment-path', type=click.Path(), required=True)
@click.option('--restore-path', type=click.Path())
def main(config_path, **kwargs):
    config = load_config(
        config_path,
        **kwargs)

    transform = T.Compose([
        RandomTemporalFlip(),
        RandomHorizontalFlip(),
        ForEach(T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.25]),
        ]))
    ])
    data_loader = torch.utils.data.DataLoader(
        Dataset(config.dataset_path, transform=transform),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True)

    model = Model()
    model.to(DEVICE)
    if config.restore_path is not None:
        model.load_state_dict(torch.load(config.restore_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.sched.steps, gamma=0.2)

    writer = SummaryWriter(config.experiment_path)
    metrics = {
        'loss': Mean(),
    }

    for epoch in range(config.epochs):
        n = 1000
        iterator = islice(data_loader, n)

        model.train()
        for images in tqdm(iterator, total=n, desc='epoch {} training'.format(epoch)):
            images_1, targets, images_3 = [image.to(DEVICE) for image in images]

            preds, etc = model(images_1, images_3)

            loss = compute_loss(input=preds, target=targets)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            metrics['loss'].update(loss.data.cpu().numpy())

        writer.add_scalar('loss', metrics['loss'].compute_and_reset(), global_step=epoch)
        writer.add_image('target', utils.make_grid((targets * 0.25 + 0.5).clamp(0., 1.)), global_step=epoch)
        writer.add_image('pred', utils.make_grid((preds * 0.25 + 0.5).clamp(0., 1.)), global_step=epoch)
        for k in etc:
            if k.startswith('flow'):
                writer.add_image(k, utils.make_grid(flow_to_rgb(etc[k])), global_step=epoch)
            else:
                writer.add_image(k, utils.make_grid(etc[k]), global_step=epoch)

        torch.save(model.state_dict(), os.path.join(config.experiment_path, 'model.pth'))
        scheduler.step()
       

def compute_loss(input, target, eps=1e-6):
    loss = torch.sqrt((input - target)**2 + eps**2).mean((1, 2, 3))

    return loss


def flow_to_rgb(flows):
    def convert(flow):
        # use hue, saturation, value color model
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return rgb

    device = flows.device
    flows = flows.permute(0, 2, 3, 1).data.cpu().numpy()

    images = [convert(flow) for flow in flows]
    images = np.stack(images, 0)
    images = (images / 255).astype(np.float32)
    images = torch.from_numpy(images).to(device).permute(0, 3, 1, 2)

    return images


if __name__ == '__main__':
    main()
