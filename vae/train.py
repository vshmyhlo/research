import argparse
import logging
import os

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.meters import Mean

# from dataset import Dataset
from tensorboardX import SummaryWriter
from torchvision.datasets import MNIST

# from ticpfptp.format import args_to_string
# from ticpfptp.metrics import Mean
# from ticpfptp.torch import fix_seed
from tqdm import tqdm

import utils
from vae.model import Model

# TODO: deeper net
# TODO: plot distributions
# TODO: tanh for mean?
# TODO: cleanup (args, code)
# TODO: interpolate between 2 codes
# TODO: discrete decoder output


MEAN = 0.5
STD = 0.25
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-path", type=str, default="./tf_log")
    parser.add_argument("--dataset-path", type=str, default="./data/cifar")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-size", type=int, default=16)
    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    config = build_parser().parse_args()
    # logging.info(args_to_string(config))
    # fix_seed(config.seed)

    transform = T.Compose([T.ToTensor(), T.Normalize([MEAN], [STD]),])

    train_data_loader = torch.utils.data.DataLoader(
        # MNIST(config.dataset_path, download=True, transform=transform),
        torchvision.datasets.CIFAR10(
            config.dataset_path, train=True, transform=transform, download=True
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True,
    )
    eval_data_loader = torch.utils.data.DataLoader(
        # MNIST(config.dataset_path, download=True, transform=transform),
        torchvision.datasets.CIFAR10(
            config.dataset_path, train=False, transform=transform, download=True
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True,
    )

    model = Model(3, config.model_size, config.latent_size)
    model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.epochs + 1):
        train(model, opt, train_data_loader, epoch, config)
        # eval(model, eval_data_loader, epoch, config)
        torch.save(model, os.path.join(config.experiment_path, "model.pt"))


def train(model, opt, data_loader, epoch, config):
    writer = SummaryWriter(os.path.join(config.experiment_path, "train"))
    metrics = {
        "loss": Mean(),
        "kl": Mean(),
        "log_pxgz": Mean(),
    }

    model.train()
    for x, _ in tqdm(data_loader, desc="epoch {} train".format(epoch)):
        x = x.to(DEVICE)

        dist_pz = model.encoder(x)
        z = dist_pz.rsample()
        dist_px = model.decoder(z)
        loss, kl, log_pxgz = compute_loss(dist_pz, z, dist_px, x)

        opt.zero_grad()
        loss.mean().backward()
        opt.step()

        metrics["loss"].update(loss.detach())
        metrics["kl"].update(kl.detach())
        metrics["log_pxgz"].update(log_pxgz.detach())

    for k in metrics:
        writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
    writer.add_image("x", utils.make_grid(denormalize(x)), global_step=epoch)
    x_hat = dist_px.sample()
    writer.add_image("x_hat", utils.make_grid(denormalize(x_hat)), global_step=epoch)
    writer.add_image("dist_x_mean", utils.make_grid(denormalize(dist_px.mean)), global_step=epoch)

    dist_pz = torch.distributions.Normal(
        torch.zeros_like(dist_pz.mean), torch.ones_like(dist_pz.scale),
    )
    z = dist_pz.rsample()
    dist_px = model.decoder(z)
    writer.add_image(
        "prior/dist_x_mean", utils.make_grid(denormalize(dist_px.mean)), global_step=epoch
    )

    writer.flush()
    writer.close()


def eval(model, data_loader, epoch, config):
    writer = SummaryWriter(os.path.join(config.experiment_path, "eval"))
    metrics = {
        "loss": Mean(),
        "kl": Mean(),
        "log_pxgz": Mean(),
    }

    model.train()
    for x, _ in tqdm(data_loader, desc="epoch {} eval".format(epoch)):
        x = x.to(DEVICE)

        dist_z = model.encoder(x)
        z = dist_z.rsample()
        dist_x = model.decoder(z)
        loss, kl, log_pxgz = compute_loss(dist_z, z, dist_x, x)

        metrics["loss"].update(loss.detach())
        metrics["kl"].update(kl.detach())
        metrics["log_pxgz"].update(log_pxgz.detach())

    for k in metrics:
        writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
    writer.add_image("x", utils.make_grid(denormalize(x)), global_step=epoch)
    x_hat = dist_x.sample()
    writer.add_image("x_hat", utils.make_grid(denormalize(x_hat)), global_step=epoch)
    writer.add_image("dist_x_mean", utils.make_grid(denormalize(dist_x.mean)), global_step=epoch)

    writer.flush()
    writer.close()


def denormalize(input):
    return (input * STD + MEAN).clamp(0, 1)


def compute_loss(dist_qzgx, z, dist_pxgz, x):
    dist_pz = torch.distributions.Normal(
        torch.zeros_like(dist_qzgx.mean), torch.ones_like(dist_qzgx.scale),
    )

    log_qzgx = dist_qzgx.log_prob(z).sum(1)
    log_pz = dist_pz.log_prob(z).sum(1)
    log_pxgz = dist_pxgz.log_prob(x).sum((1, 2, 3))

    kl = log_qzgx - log_pz
    assert kl.size() == log_pxgz.size()
    loss = kl - log_pxgz

    return loss, kl, log_pxgz


if __name__ == "__main__":
    main()
