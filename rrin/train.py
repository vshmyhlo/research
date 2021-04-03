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
from rrin.dataset import ForEach, RandomHorizontalFlip, RandomTemporalFlip, Vimeo90kDataset
from rrin.model import Model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option("--config-path", type=click.Path(), required=True)
@click.option("--dataset-path", type=click.Path(), required=True)
@click.option("--experiment-path", type=click.Path(), required=True)
@click.option("--restore-path", type=click.Path())
def main(config_path, **kwargs):
    config = load_config(config_path, **kwargs)

    eval_transform = ForEach(
        T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.25]),
            ]
        )
    )
    train_transform = T.Compose(
        [
            RandomTemporalFlip(),
            RandomHorizontalFlip(),
            eval_transform,
        ]
    )
    train_data_loader = torch.utils.data.DataLoader(
        Vimeo90kDataset(config.dataset_path, subset="train", transform=train_transform),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True,
    )
    eval_data_loader = torch.utils.data.DataLoader(
        Vimeo90kDataset(config.dataset_path, subset="test", transform=eval_transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        drop_last=False,
    )
    # vis_data_loader = torch.utils.data.DataLoader(
    #     MiddleburyDataset(
    #         os.path.join(config.dataset_path, '..', 'middlebury-eval'), video='Dumptruck', transform=eval_transform),
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=os.cpu_count(),
    #     drop_last=False)

    model = Model()
    model.to(DEVICE)
    if config.restore_path is not None:
        model.load_state_dict(torch.load(config.restore_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.sched.steps, gamma=0.2)

    train_writer = SummaryWriter(os.path.join(config.experiment_path, "train"))
    eval_writer = SummaryWriter(os.path.join(config.experiment_path, "eval"))

    for epoch in range(config.epochs):
        model.train()
        train_epoch(train_data_loader, model, optimizer, train_writer, epoch)

        model.eval()
        with torch.no_grad():
            eval_epoch(eval_data_loader, model, eval_writer, epoch)
            # vis_epoch(vis_data_loader, model, eval_writer, epoch)

        torch.save(model.state_dict(), os.path.join(config.experiment_path, "model.pth"))
        scheduler.step()


def train_epoch(data_loader, model, optimizer, writer, epoch):
    metrics = {
        "loss": Mean(),
    }

    n = 1000
    iterator = islice(data_loader, n)
    for images in tqdm(iterator, total=n, desc="epoch {} training".format(epoch)):
        images_1, targets, images_3 = [image.to(DEVICE) for image in images]

        preds, etc = model(images_1, images_3)

        loss = compute_loss(input=preds, target=targets)

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        metrics["loss"].update(loss.data.cpu().numpy())

    writer.add_scalar("loss", metrics["loss"].compute_and_reset(), global_step=epoch)
    writer.add_image("target", utils.make_grid(denormalize(targets)), global_step=epoch)
    writer.add_image("pred", utils.make_grid(denormalize(preds)), global_step=epoch)
    with torch.no_grad():
        video = make_video(model, images_1, images_3)
    writer.add_video("video/interp", video, fps=5, global_step=epoch)
    for k in etc:
        if k.startswith("flow"):
            writer.add_image(k, utils.make_grid(flow_to_rgb(etc[k])), global_step=epoch)
        else:
            writer.add_image(k, utils.make_grid(etc[k]), global_step=epoch)
    writer.flush()


def eval_epoch(data_loader, model, writer, epoch):
    metrics = {
        "loss": Mean(),
    }

    for images in tqdm(data_loader, desc="epoch {} evaluation".format(epoch)):
        images_1, targets, images_3 = [image.to(DEVICE) for image in images]

        preds, etc = model(images_1, images_3)

        loss = compute_loss(input=preds, target=targets)

        metrics["loss"].update(loss.data.cpu().numpy())

    writer.add_scalar("loss", metrics["loss"].compute_and_reset(), global_step=epoch)
    writer.add_image("target", utils.make_grid(denormalize(targets)), global_step=epoch)
    writer.add_image("pred", utils.make_grid(denormalize(preds)), global_step=epoch)
    with torch.no_grad():
        video = make_video(model, images_1, images_3)
    writer.add_video("video/interp", video, fps=5, global_step=epoch)
    for k in etc:
        if k.startswith("flow"):
            writer.add_image(k, utils.make_grid(flow_to_rgb(etc[k])), global_step=epoch)
        else:
            writer.add_image(k, utils.make_grid(etc[k]), global_step=epoch)
    writer.flush()


def make_video(model, images_1, images_3):
    interp_video = []

    interp_video.append(torch.zeros_like(images_1))
    interp_video.append(denormalize(images_1))
    for t in np.arange(0.1, 1.0, 0.1):
        preds, etc = model(images_1, images_3, t=t)
        interp_video.append(denormalize(preds))
    interp_video.append(denormalize(images_3))
    interp_video.append(torch.zeros_like(images_3))

    interp_video = [utils.make_grid(images) for images in interp_video]
    interp_video = torch.stack(interp_video, 0).unsqueeze(0)

    return interp_video


def vis_epoch(data_loader, model, writer, epoch):
    raw_video = []
    interp_video = []

    for images in tqdm(data_loader, desc="epoch {} visualization".format(epoch)):
        images_1, images_3 = [image.to(DEVICE) for image in images]

        raw_video.append(denormalize(images_1))
        interp_video.append(denormalize(images_1))

        for t in np.arange(0.1, 1.0, 0.1):
            if t < 0.5:
                raw_video.append(denormalize(images_1))
            else:
                raw_video.append(denormalize(images_3))

            preds, etc = model(images_1, images_3, t=t)
            interp_video.append(denormalize(preds))

    raw_video.append(denormalize(images_3))
    interp_video.append(denormalize(images_3))

    raw_video = torch.cat(raw_video, 0)
    interp_video = torch.cat(interp_video, 0)

    writer.add_video("video/raw", raw_video.unsqueeze(0), fps=12, global_step=epoch)
    writer.add_video("video/interp", interp_video.unsqueeze(0), fps=12, global_step=epoch)
    writer.flush()


def compute_loss(input, target, eps=1e-6):
    loss = torch.sqrt((input - target) ** 2 + eps ** 2).mean((1, 2, 3))

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


def denormalize(input):
    # return input.clamp(0., 1.)
    return (input * 0.25 + 0.5).clamp(0.0, 1.0)


if __name__ == "__main__":
    main()
