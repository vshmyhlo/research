import os

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.config import load_config
from all_the_tools.metrics import Mean, Last
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils
from gan.model_v2 import Discriminator, Generator
from gan.model_v2.generator import ZeroBlock
from gan.modules import AdditiveNoise
from transforms import Resettable

# TODO: spherical z
# TODO: spherical interpolation
# TODO: norm z


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--dataset-path', type=click.Path(), required=True)
@click.option('--experiment-path', type=click.Path(), required=True)
@click.option('--restore-path', type=click.Path())
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(config_path, **kwargs):
    config = load_config(
        config_path,
        **kwargs)

    transform, update_transform = build_transform()

    if config.dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(config.dataset_path, transform=transform, download=True)
    elif config.dataset == 'celeba':
        dataset = torchvision.datasets.ImageFolder(config.dataset_path, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        drop_last=True)

    model = nn.ModuleDict({
        'discriminator': Discriminator(
            config.image_size),
        'generator': Generator(
            config.image_size, config.latent_size),
    })
    model.to(DEVICE)
    model.apply(weight_init)
    if config.restore_path is not None:
        model.load_state_dict(torch.load(config.restore_path))

    discriminator_opt = torch.optim.Adam(
        model.discriminator.parameters(), lr=config.opt.lr, betas=(0., 0.99), eps=1e-8)
    generator_opt = torch.optim.Adam(
        model.generator.parameters(), lr=config.opt.lr, betas=(0., 0.99), eps=1e-8)

    noise_dist = torch.distributions.Normal(0, 1)

    writer = SummaryWriter(config.experiment_path)
    metrics = {
        'loss/discriminator': Mean(),
        'loss/generator': Mean(),
        'level': Last(),
        'alpha': Last(),
    }

    for epoch in range(1, config.epochs + 1):
        model.train()

        level, _ = compute_level(
            epoch - 1, config.epochs, 0, len(data_loader), config.image_size, config.grow_min_level)
        update_transform(int(4 * 2**level))

        for i, (real, _) in enumerate(tqdm(data_loader, desc='epoch {} training'.format(epoch))):
            _, a = compute_level(
                epoch - 1, config.epochs, i, len(data_loader), config.image_size, config.grow_min_level)

            real = real.to(DEVICE)

            # discriminator ############################################################################################
            discriminator_opt.zero_grad()

            # real
            scores = model.discriminator(real, level=level, a=a)
            loss = F.softplus(-scores)
            loss.mean().backward()
            loss_real = loss

            # fake
            noise = noise_dist.sample((config.batch_size, config.latent_size)).to(DEVICE)
            fake = model.generator(noise, level=level, a=a)
            assert real.size() == fake.size()
            scores = model.discriminator(fake, level=level, a=a)
            loss = F.softplus(scores)
            loss.mean().backward()
            loss_fake = loss

            discriminator_opt.step()
            metrics['loss/discriminator'].update((loss_real + loss_fake).data.cpu().numpy())

            # generator ################################################################################################
            generator_opt.zero_grad()

            # fake
            noise = noise_dist.sample((config.batch_size, config.latent_size)).to(DEVICE)
            fake = model.generator(noise, level=level, a=a)
            assert real.size() == fake.size()
            scores = model.discriminator(fake, level=level, a=a)
            loss = F.softplus(-scores)
            loss.mean().backward()

            generator_opt.step()
            metrics['loss/generator'].update(loss.data.cpu().numpy())

            metrics['level'].update(level)
            metrics['alpha'].update(a)

        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image('real', utils.make_grid((real + 1) / 2), global_step=epoch)
        writer.add_image('fake', utils.make_grid((fake + 1) / 2), global_step=epoch)

        torch.save(model.state_dict(), os.path.join(config.experiment_path, 'model_{}.pth'.format(epoch)))


def compute_level(epoch, max_epochs, step, max_steps, image_size, min_level):
    num_levels = np.log2(image_size / 4).astype(np.int32) - min_level + 1
    assert max_epochs % num_levels == 0
    epochs_per_level = max_epochs // num_levels

    # epoch = np.arange(max_epochs)
    level = epoch // epochs_per_level + min_level
    # print(np.bincount(level))

    # step = np.arange(max_steps)
    prog = step / max_steps
    prog = ((epoch % epochs_per_level) + prog) / epochs_per_level
    a = np.minimum(prog * 2, 1.)
    # print(a.round(2))

    return level, a


def build_transform():
    def update_transform(image_size):
        print('image_size {}'.format(image_size))
        resize.reset(image_size)
        crop.reset(image_size)

    def to_rgb(input):
        return input.convert('RGB')

    resize = Resettable(T.Resize)
    crop = Resettable(T.CenterCrop)
    transform = T.Compose([
        T.Lambda(to_rgb),
        resize,
        crop,
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    return transform, update_transform


def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d,)):
        torch.nn.init.normal_(m.weight, 0., 1.)
        torch.nn.init.constant_(m.bias, 0.)
    elif isinstance(m, (AdditiveNoise,)):
        torch.nn.init.constant_(m.weight, 0.)
    elif isinstance(m, (ZeroBlock,)):
        torch.nn.init.constant_(m.input, 1.)


if __name__ == '__main__':
    main()
