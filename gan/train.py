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
from tensorboardX import SummaryWriter
from ticpfptp.metrics import Mean
from tqdm import tqdm

import utils
from gan.model import Discriminator, Generator
from transforms import Resettable

# TODO: spherical z
# TODO: spherical interpolation
# TODO: norm z


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

    transform, update_transform = build_transform()

    if config.dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(config.dataset_path, transform=transform, download=True)
    elif config.dataset == 'celeba':
        dataset = torchvision.datasets.ImageFolder(config.dataset_path, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True)

    model = nn.ModuleDict({
        'discriminator': Discriminator(
            config.image_size),
        'generator': Generator(
            config.image_size, config.latent_size),
    })
    model.to(DEVICE)
    model.apply(weights_init)
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
        'loss/generator': Mean()
    }

    for epoch in range(1, config.epochs + 1):
        model.train()

        ###
        p = (epoch - 1) / config.epochs
        level = np.log2(config.image_size / 4)
        level = np.floor((level + 1) * p).astype(np.int32)

        steps = np.linspace(0, 1, len(data_loader))
        alphas = np.minimum(steps * 2, 1.)

        update_transform(int(4 * 2**level))
        ###

        for i, (real, _) in enumerate(tqdm(data_loader, desc='epoch {} training'.format(epoch))):
            real = real.to(DEVICE)

            # discriminator ############################################################################################
            discriminator_opt.zero_grad()

            # real
            scores = model.discriminator(real, level=level, a=alphas[i])
            loss = F.softplus(-scores)
            loss.mean().backward()
            loss_real = loss

            # fake
            noise = noise_dist.sample((config.batch_size, config.latent_size)).to(DEVICE)
            fake = model.generator(noise, level=level, a=alphas[i])
            assert real.size() == fake.size()
            scores = model.discriminator(fake, level=level, a=alphas[i])
            loss = F.softplus(scores)
            loss.mean().backward()
            loss_fake = loss

            discriminator_opt.step()
            metrics['loss/discriminator'].update((loss_real + loss_fake).data.cpu().numpy())

            # generator ################################################################################################
            generator_opt.zero_grad()

            # fake
            noise = noise_dist.sample((config.batch_size, config.latent_size)).to(DEVICE)
            fake = model.generator(noise, level=level, a=alphas[i])
            assert real.size() == fake.size()
            scores = model.discriminator(fake, level=level, a=alphas[i])
            loss = F.softplus(-scores)
            loss.mean().backward()

            generator_opt.step()
            metrics['loss/generator'].update(loss.data.cpu().numpy())

        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image('real', utils.make_grid((real + 1) / 2), global_step=epoch)
        writer.add_image('fake', utils.make_grid((fake + 1) / 2), global_step=epoch)

        torch.save(model.state_dict(), os.path.join(config.experiment_path, 'model.pth'))


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


def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight.data, 0., 0.02)
    elif isinstance(m, (nn.BatchNorm2d,)):
        torch.nn.init.normal_(m.weight.data, 1., 0.02)
        torch.nn.init.constant_(m.bias.data, 0.)


if __name__ == '__main__':
    main()
