import copy
import os

import click
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.config import load_config
from all_the_tools.meters import Concat, Mean
from all_the_tools.torch.data import ChunkedDataLoader
from all_the_tools.torch.utils import ModuleEMA
from tqdm import tqdm

from datasets.image_folder import ImageFolderDataset
from gan.losses import BinaryCrossEntropyLoss, LogisticNSLoss, NonSatLogisticLoss, WassersteinLoss
from precision_recall import plot_pr_curve, precision_recall_auc
from stylegan.model.dsc import Dsc
from stylegan.model.gen import Gen
from summary_writers.file_system import SummaryWriter
from utils import compute_nrow, zero_grad_and_step

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# TODO: test style-mixing code
# TODO: review minibatch-std
# TODO: check style mixing layer ordering
# TODO: check https://github.com/NVlabs/stylegan2-ada/blob/main/training/networks.py
# TODO: larger number of channels
# TODO: save weights and opt state
# TODO: path-length regularization
# TODO: is it ok to use same real for both gen and dsc?
# TODO: viz rgb contribution
# TODO: check g_main
# TODO: check that all defined layers are used
# TODO: mb discrimination vs mb stddev (progan)
# TODO: EMA weight
# TODO: 512 z space on a hypersphere
# TODO: 8-layer 512 mlp
# TODO: rename affine, it's w->affine->style transform
# TODO: rename style->w (not everywhere)
# TODO: init and weight scale/gain depends on activation
# TODO: up and downsapling via bilinear interpolation
# TODO: check that starts from 4x4 and ends at 4x4
# TODO: drift regularization? (progan)
# TODO: lsgan discriminator noise hack (progan)
# TODO: adain removes any statistics and adds statistics from the style before next cov
# TODO: style mixing regularization (stylegan)
# TODO: reg interval and epoch size
# TODO: bilinear sampling, which we implement by filtering the activations after each upsampling layer and before each downsampling layer
# TODO: slower channel size growing
# TODO: conv->bias,noise->(relu)->upsample->conv
# TODO: The activation function (leaky ReLU) is always applied right after adding the bias
# TODO: review group conv usage
# TODO: downsampling stride
# TODO: use truncation trick
# TODO: upsample_conv, conv_downsample

"""
To avoid having to account for the
activation function in Equation 3, we scale our activation
functions so that they retain the expected signal variance.
"""


"""
We deviate from the current trend of careful weight initialization, and instead use a trivial N (0, 1)
initialization and then explicitly scale the weights at runtime. To be precise, we set wˆi = wi/c,
where wi are the weights and c is the per-layer normalization constant from He’s initializer
"""


"""
We initialize all weights of the convolutional,
fully-connected, and affine transform layers using N (0, 1).
The constant input in synthesis network is initialized to one.
The biases and noise scaling factors are initialized to zero,
except for the biases associated with ys that we initialize to
one. (stylegan)
"""

"""
To compensate for the fact that we now perform
k+1 training iterations instead of k, we adjust the optimizer
hyperparameters λ
0 = c · λ, β
0
1 = (β1)
c
, and β
0
2 = (β2)
c
,
where c = k/(k + 1). We also multiply the regularization
term by k to balance the overall magnitude of its gradients.
We use k = 16 for the discriminator and k = 8 for the
generator.
"""


# TODO: fake vs fake-ema visualization
# TODO: bilinear upsample
# TODO: output std of fakes
# TODO: use minibatch stddev to stabilize training
# TODO: Dima's idea on set-transformer
# TODO: spectral norm
# TODO: generator's weights moving average


@click.command()
@click.option("--config-path", type=click.Path(), required=True)
@click.option("--experiment-path", type=click.Path(), required=True)
@click.option("--num-workers", type=click.INT, default=os.cpu_count())
@click.option("--debug", is_flag=True)
def main(config_path, **kwargs):
    config = load_config(config_path, **kwargs)

    noise_dist = torch.distributions.Normal(0, 1)
    gen = Gen(
        image_size=config.image_size,
        base_channels=config.gen.base_channels,
        max_channels=config.gen.max_channels,
        z_channels=config.noise_size,
        noise_dist=noise_dist,
    ).to(DEVICE)
    dsc = Dsc(
        image_size=config.image_size,
        base_channels=config.dsc.base_channels,
        max_channels=config.dsc.max_channels,
    ).to(DEVICE)
    gen_ema = copy.deepcopy(gen)
    ema = ModuleEMA(gen_ema, 0.998)

    opt_gen = build_optimizer(gen.parameters(), config)
    opt_dsc = build_optimizer(dsc.parameters(), config)

    state = torch.load(os.path.join(config.experiment_path, "checkpoint.pth"))
    dsc.load_state_dict(state["dsc"])
    gen.load_state_dict(state["gen"])
    gen_ema.load_state_dict(state["gen_ema"])
    opt_gen.load_state_dict(state["opt_gen"])
    opt_dsc.load_state_dict(state["opt_dsc"])

    dataset = build_dataset(config)
    print("dataset size: {}".format(len(dataset)))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader = ChunkedDataLoader(data_loader, config.batches_in_epoch)

    noise_fixed = noise_dist.sample((8 ** 2, config.noise_size)).to(DEVICE)
    dsc_compute_loss, gen_compute_loss = build_loss(config)

    writer = SummaryWriter(config.experiment_path)
    for epoch in range(1, config.num_epochs + 1):
        metrics = {
            "dsc/loss": Mean(),
            "gen/loss": Mean(),
        }
        dsc_logits = Concat()
        dsc_targets = Concat()

        gen.train()
        dsc.train()
        gen_ema.train()
        for batch_i, real in enumerate(
            tqdm(
                data_loader,
                desc="{}/{}".format(epoch, config.num_epochs),
                disable=config.debug,
                smoothing=0.1,
            ),
            1,
        ):
            real = real.to(DEVICE)

            # generator: train
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

            # generator: update moving average
            ema.update(gen)

            # discriminator: train
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

            # discriminator: regularize
            if batch_i % config.dsc.reg_interval == 0:
                with zero_grad_and_step(opt_dsc):
                    # R1 regularization
                    real = real.detach().requires_grad_(True)
                    logits = dsc(real)
                    (real_grads,) = torch.autograd.grad(
                        outputs=[logits.sum()], inputs=[real], create_graph=True, only_inputs=True
                    )
                    r1_penalty = real_grads.square().sum([1, 2, 3])
                    loss_r1 = r1_penalty * (config.r1_gamma * 0.5) * config.dsc.reg_interval
                    loss_r1.mean().backward()

        dsc.eval()
        gen.eval()
        gen_ema.eval()
        with torch.no_grad():
            fake = gen(noise_fixed)
            fake_ema = gen_ema(noise_fixed)
            real, fake, fake_ema = [denormalize(x).clamp(0, 1) for x in [real, fake, fake_ema]]

            dsc_logits = dsc_logits.compute_and_reset().data.cpu().numpy()
            dsc_targets = dsc_targets.compute_and_reset().data.cpu().numpy()

            metrics = {k: metrics[k].compute_and_reset() for k in metrics}
            metrics["dsc/ap"] = precision_recall_auc(input=dsc_logits, target=dsc_targets)
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
            writer.add_image(
                "fake_ema",
                torchvision.utils.make_grid(fake_ema, nrow=compute_nrow(fake_ema)),
                global_step=epoch,
            )
            torch.save(
                {
                    "gen": gen.state_dict(),
                    "gen_ema": gen_ema.state_dict(),
                    "dsc": dsc.state_dict(),
                    "opt_gen": opt_gen.state_dict(),
                    "opt_dsc": opt_dsc.state_dict(),
                },
                os.path.join(config.experiment_path, "checkpoint.pth"),
            )

    writer.flush()
    writer.close()


def build_optimizer(parameters, config):
    if config.opt.type == "adam":
        return torch.optim.Adam(parameters, **config.opt.args)
    elif config.opt.type == "rmsprop":
        return torch.optim.RMSprop(parameters, **config.opt.args)
    else:
        raise ValueError("invalid config.opt.type {}".format(config.opt.type))


def denormalize(input):
    return input * 0.5 + 0.5


def build_loss(config):
    def build(loss_type):
        if loss_type == "bce":
            return BinaryCrossEntropyLoss()
        elif loss_type == "nslog":
            return NonSatLogisticLoss()
        elif loss_type == "logns":
            return LogisticNSLoss()
        elif loss_type == "wass":
            return WassersteinLoss()
        else:
            raise ValueError("invalid loss type {}".format(loss_type))

    return build(config.dsc.loss), build(config.gen.loss)


def build_dataset(config):
    if config.dataset == "ffhq":
        transform = T.Compose(
            [
                T.Resize(config.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        return ImageFolderDataset("../ffhq", transform=transform)
    elif config.dataset == "wikiart":
        transform = T.Compose(
            [
                T.Resize(config.image_size),
                T.RandomHorizontalFlip(),
                T.RandomCrop(config.image_size),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        return torch.utils.data.ConcatDataset(
            [
                # ImageFolderDataset("./data/wikiart/resized/abstract", transform=transform),
                # ImageFolderDataset("./data/wikiart/resized/cityscape", transform=transform),
                # ImageFolderDataset("./data/wikiart/resized/figurative", transform=transform),
                # ImageFolderDataset("./data/wikiart/resized/landscape", transform=transform),
                # ImageFolderDataset("./data/wikiart/resized/portrait", transform=transform),
                ImageFolderDataset("./data/wikiart-resized", transform=transform),
            ]
        )
    else:
        raise ValueError("invalid config.dataset {}".format(config.dataset))

    # dataset = torchvision.datasets.MNIST(
    #     "./data/mnist", train=True, transform=transform, download=True
    # )
    # dataset = torchvision.datasets.CelebA(
    #     "./data/celeba", split="all", transform=transform, download=True
    # )


if __name__ == "__main__":
    main()