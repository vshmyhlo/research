import os
import time

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.meters import Mean
from torchvision.datasets import MNIST, SVHN
from tqdm import tqdm

from cycle_gan.model.dsc import Dsc
from cycle_gan.model.gen import Gen
from gan.losses import LeastSquaresLoss
from summary_writers.file_system import SummaryWriter
from utils import stack_images

# TODO: review optimization process


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@click.command()
@click.option("--experiment-path", type=click.Path(), required=True)
@click.option("--workers", type=click.INT, default=os.cpu_count())
def main(experiment_path, workers):
    epochs = 100

    transform = T.Compose(
        [
            T.Resize(32),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
            T.Lambda(lambda x: x if x.size(0) == 3 else x.repeat(3, 1, 1)),
        ]
    )

    data_loader_a = torch.utils.data.DataLoader(
        MNIST("./data/mnist", train=True, transform=transform, download=True),
        batch_size=64,
        num_workers=workers,
        shuffle=True,
        drop_last=True,
    )
    data_loader_b = torch.utils.data.DataLoader(
        SVHN("./data/svhn", split="train", transform=transform, download=True),
        batch_size=64,
        num_workers=workers,
        shuffle=True,
        drop_last=True,
    )

    dsc = nn.ModuleDict(
        {
            "a": Dsc().to(DEVICE),
            "b": Dsc().to(DEVICE),
        }
    )
    gen = nn.ModuleDict(
        {
            "a_b": Gen().to(DEVICE),
            "b_a": Gen().to(DEVICE),
        }
    )

    dsc.train()
    gen.train()

    LR = 0.0002
    dsc_opt = torch.optim.Adam(dsc.parameters(), LR, betas=(0.5, 0.999))
    gen_opt = torch.optim.Adam(gen.parameters(), LR, betas=(0.5, 0.999))

    metrics = {
        "gen/loss": Mean(),
        "dsc/loss": Mean(),
    }

    compute_loss = LeastSquaresLoss()

    writer = SummaryWriter(experiment_path)
    for epoch in range(1, epochs + 1):
        for batch_i, ((real_a, _), (real_b, _)) in tqdm(
            enumerate(zip(data_loader_a, data_loader_b)),
            desc="epoch {}".format(epoch),
            total=min(len(data_loader_a), len(data_loader_b)),
        ):
            real_a, real_b = real_a.to(DEVICE), real_b.to(DEVICE)

            if batch_i % 2 == 0:
                # train discriminator ==================================================================================

                with torch.no_grad():
                    intr_a = gen.b_a(real_b)
                    intr_b = gen.a_b(real_a)

                loss = 0.5 * mean(
                    [
                        compute_loss(dsc.a(real_a), True).mean(),
                        compute_loss(dsc.b(real_b), True).mean(),
                        compute_loss(dsc.a(intr_a), False).mean(),
                        compute_loss(dsc.b(intr_b), False).mean(),
                    ]
                )

                dsc_opt.zero_grad()
                loss.mean().backward()
                dsc_opt.step()

                metrics["dsc/loss"].update(loss.detach())

            else:
                # train generator ======================================================================================

                intr_a = gen.b_a(real_b)
                intr_b = gen.a_b(real_a)
                back_a = gen.b_a(intr_b)
                back_b = gen.a_b(intr_a)

                loss = mean(
                    [
                        10 * F.l1_loss(input=back_a, target=real_a, reduction="mean"),
                        10 * F.l1_loss(input=back_b, target=real_b, reduction="mean"),
                        compute_loss(dsc.a(intr_a), True).mean(),
                        compute_loss(dsc.b(intr_b), True).mean(),
                    ]
                )

                gen_opt.zero_grad()
                loss.mean().backward()
                gen_opt.step()

                metrics["gen/loss"].update(loss.detach())
                images = [real_a, real_b, intr_a, intr_b, back_a, back_b]

        with torch.no_grad():
            real_a, real_b, intr_a, intr_b, back_a, back_b = [
                denormalize(x[: 8 ** 2]).clamp(0, 1) for x in images
            ]
            for k in metrics:
                writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
            a_b, nrow = stack_images([real_a, intr_b, back_a])
            writer.add_image(
                "a_b",
                torchvision.utils.make_grid(a_b, nrow=nrow),
                global_step=epoch,
            )
            b_a, nrow = stack_images([real_b, intr_a, back_b])
            writer.add_image(
                "b_a",
                torchvision.utils.make_grid(b_a, nrow=nrow),
                global_step=epoch,
            )

            writer.flush()


def denormalize(input):
    return input * 0.5 + 0.5


def mean(seq):
    return sum(seq) / len(seq)


if __name__ == "__main__":
    main()
