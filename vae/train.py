import argparse
import logging
import os

import torch
import torch.nn.functional as F
import torch.utils.data
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
from vae.model import Decoder, Encoder

# TODO: spherical z
# TODO: spherical interpolation
# TODO: norm z
# TODO: cleanup (args, code)


MEAN = 0.5
STD = 0.25


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-path", type=str, default="./tf_log")
    parser.add_argument("--dataset-path", type=str, default="./data/mnist")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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

    data_loader = torch.utils.data.DataLoader(
        MNIST(config.dataset_path, download=True, transform=transform),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(config.model_size, config.latent_size)
    decoder = Decoder(config.model_size, config.latent_size)
    encoder.to(device)
    decoder.to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=config.learning_rate
    )

    writer = SummaryWriter(config.experiment_path)
    metrics = {
        "loss": Mean(),
        "kl": Mean(),
        "log_pxgz": Mean(),
    }

    for epoch in range(1, config.epochs + 1):
        encoder.train()
        decoder.train()
        for x, _ in tqdm(data_loader, desc="epoch {} training".format(epoch)):
            x = x.to(device)

            dist_z = encoder(x)
            z = dist_z.rsample()
            dist_x = decoder(z)
            loss, kl, log_pxgz = compute_loss(dist_z, z, dist_x, x)

            opt.zero_grad()
            loss.mean().backward()
            opt.step()

            metrics["loss"].update(loss.detach())
            metrics["kl"].update(kl.detach())
            metrics["log_pxgz"].update(log_pxgz.detach())

        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image("x", utils.make_grid(denormalize(x)), global_step=epoch)
        x_hat = dist_x.sample()
        writer.add_image("x_hat", utils.make_grid(denormalize(x_hat)), global_step=epoch)
        writer.add_image(
            "dist_x_mean", utils.make_grid(denormalize(dist_x.mean)), global_step=epoch
        )


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
    loss = kl - log_pxgz

    assert kl.size() == log_pxgz.size()

    return loss, kl, log_pxgz


if __name__ == "__main__":
    main()
