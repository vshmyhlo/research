import copy
import os

import click
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.config import load_config
from all_the_tools.meters import Concat, Mean
from metrics import plot_pr_curve, precision_recall_auc
from tqdm import tqdm

from datasets.image_folder import ImageFolderDataset
from gan.losses import BinaryCrossEntropyLoss, LogisticNSLoss, NonSatLogisticLoss, WassersteinLoss
from stylegan.model.dsc import Dsc
from stylegan.model.gen import Gen
from summary_writers.file_system import SummaryWriter
from utils import ModuleEMA, clip_parameters, compute_nrow, weighted_sum, zero_grad_and_step

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# TODO: fake vs fake-ema visualization
# TODO: bilinear upsample
# TODO: output std of fakes
# TODO: use minibatch stddev to stabilize training
# TODO: Dima's idea on set-transformer
# TODO: spectral norm
# TODO: generator's weights moving average


@click.command()
@click.option("--config-path", type=click.Path(), required=True)
@click.option("--num-workers", type=click.INT, default=os.cpu_count())
@click.option("--debug", is_flag=True)
def main(config_path, **kwargs):
    config = load_config(config_path, **kwargs)

    gen = Gen(
        image_size=config.image_size,
        image_channels=3,
        base_channels=config.gen.base_channels,
        z_channels=config.noise_size,
    ).to(DEVICE)
    dsc = Dsc(
        image_size=config.image_size,
        image_channels=3,
        base_channels=config.dsc.base_channels,
    ).to(DEVICE)
    # gen_ema = gen
    # gen_ema = copy.deepcopy(gen)
    # ema = EMA(gen_ema, 0.99)

    gen.train()
    dsc.train()
    # gen_ema.train()

    opt_gen = build_optimizer(gen.parameters(), config)
    opt_dsc = build_optimizer(dsc.parameters(), config)

    transform = T.Compose(
        [
            T.Resize(config.image_size),
            T.RandomCrop(config.image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

    # dataset = torchvision.datasets.MNIST(
    #     "./data/mnist", train=True, transform=transform, download=True
    # )
    # dataset = torchvision.datasets.CelebA(
    #     "./data/celeba", split="all", transform=transform, download=True
    # )
    dataset = ImageFolderDataset("./data/wikiart/resized/landscape", transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    noise_dist = torch.distributions.Normal(0, 1)
    dsc_compute_loss, gen_compute_loss = build_loss(config)

    writer = SummaryWriter("./log")
    for epoch in range(1, config.num_epochs + 1):
        metrics = {
            "dsc/loss": Mean(),
            "gen/loss": Mean(),
        }
        dsc_logits = Concat()
        dsc_targets = Concat()

        for batch_index, real in enumerate(
            tqdm(data_loader, desc="{}/{}".format(epoch, config.num_epochs), disable=config.debug)
        ):
            real = real.to(DEVICE)

            # train discriminator
            with zero_grad_and_step(opt_dsc):
                if config.debug:
                    print("dsc")
                noise = noise_dist.sample((real.size(0), config.noise_size)).to(DEVICE)
                with torch.no_grad():
                    fake = gen(noise)
                    assert (
                        fake.size() == real.size()
                    ), "fake size {} does not match real size {}".format(fake.size(), real.size())

                # dsc real
                logits = dsc(real)
                loss = dsc_compute_loss(logits, True)
                loss.mean().backward()
                metrics["dsc/loss"].update(loss.detach())

                dsc_logits.update(logits.detach())
                dsc_targets.update(torch.ones_like(logits))

                # dsc fake
                logits = dsc(fake.detach())
                loss = dsc_compute_loss(logits, False)
                loss.mean().backward()
                metrics["dsc/loss"].update(loss.detach())

                dsc_logits.update(logits.detach())
                dsc_targets.update(torch.zeros_like(logits))

                if (batch_index + 1) % 8 != 0:
                    # r1
                    r1_gamma = 10
                    real = real.detach().requires_grad_(True)
                    logits = dsc(real)
                    (r1_grads,) = torch.autograd.grad(
                        outputs=[logits.sum()], inputs=[real], create_graph=True, only_inputs=True
                    )
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_r1 = r1_penalty * (r1_gamma / 2) * 8
                    loss_r1.mean().backward()

            if config.dsc.weight_clip is not None:
                clip_parameters(dsc, config.dsc.weight_clip)

            if (batch_index + 1) % config.dsc.num_steps != 0:
                continue

            # train generator
            with zero_grad_and_step(opt_gen):
                if config.debug:
                    print("gen")
                noise = noise_dist.sample((real.size(0), config.noise_size)).to(DEVICE)
                fake = gen(noise)
                assert (
                    fake.size() == real.size()
                ), "fake size {} does not match real size {}".format(fake.size(), real.size())

                # gen fake
                logits = dsc(fake)
                loss = gen_compute_loss(logits, True)
                loss.mean().backward()
                metrics["gen/loss"].update(loss.detach())

                # update moving average
                # ema.update(gen)

        with torch.no_grad():
            real, fake = [(x[: 4 ** 2] * 0.5 + 0.5).clamp(0, 1) for x in [real, fake]]

            dsc_logits = dsc_logits.compute_and_reset().data.cpu().numpy()
            dsc_targets = dsc_targets.compute_and_reset().data.cpu().numpy()

            metrics = {k: metrics[k].compute_and_reset() for k in metrics}
            metrics["ap"] = precision_recall_auc(input=dsc_logits, target=dsc_targets)
            for k in metrics:
                writer.add_scalar(k, metrics[k], global_step=epoch)
            writer.add_figure(
                "dsc/pr_curve",
                plot_pr_curve(input=dsc_logits, target=dsc_targets),
                global_step=epoch,
            )
            writer.add_image(
                "real",
                torchvision.utils.make_grid(real, nrow=compute_nrow(real)),
                global_step=epoch,
            )
            writer.add_image(
                "fake",
                torchvision.utils.make_grid(fake, nrow=compute_nrow(fake)),
                global_step=epoch,
            )
            # writer.add_image(
            #     "fake_ema",
            #     torchvision.utils.make_grid(fake, nrow=compute_nrow(fake)),
            #     global_step=epoch,
            # )

    writer.flush()
    writer.close()


def build_optimizer(parameters, config):
    if config.opt.type == "adam":
        return torch.optim.Adam(parameters, **config.opt.args)
    elif config.opt.type == "rmsprop":
        return torch.optim.RMSprop(parameters, **config.opt.args)
    else:
        raise ValueError("invalid config.opt.type {}".format(config.opt.type))


def build_loss(config):
    def build(loss_type):
        if loss_type == "bce":
            return BinaryCrossEntropyLoss()
        elif loss_type == "nslog":
            return NonSatLogisticLoss()
        elif loss_type == "sp":
            return LogisticNSLoss()
        elif loss_type == "wass":
            return WassersteinLoss()
        else:
            raise ValueError("invalid loss type {}".format(loss_type))

    return build(config.dsc.loss), build(config.gen.loss)


if __name__ == "__main__":
    main()
