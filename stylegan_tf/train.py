import os

import click
import tensorflow as tf
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.config import load_config
from all_the_tools.metrics import Mean
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils
from stylegan_tf.model import G_main as generator, D_stylegan2 as discriminator
from transforms import Resettable

tf.compat.v1.disable_eager_execution()


# TODO: spherical z
# TODO: spherical interpolation
# TODO: norm z


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

    ins, outs = build_graph(config)

    writer = SummaryWriter(config.experiment_path)
    metrics = {
        'd/loss': Mean(),
        'g/loss': Mean(),
    }

    with tf.Session() as sess:
        for epoch in range(1, config.epochs + 1):
            for i, (real, _) in enumerate(tqdm(data_loader, desc='epoch {} training'.format(epoch))):
                # discriminator ########################################################################################

                _, d_loss = sess.run(
                    [outs['d']['update'], outs['d']['loss']])

                print(d_loss.shape)
                metrics['d/loss'].update(d_loss)

                # generator ############################################################################################

                _, g_loss = sess.run(
                    [outs['g']['update'], outs['g']['loss']])

                print(g_loss.shape)
                metrics['g/loss'].update(g_loss)

            fake = sess.run(outs['fake'])

            for k in metrics:
                writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
            writer.add_image('real', utils.make_grid((real + 1) / 2), global_step=epoch)
            writer.add_image('fake', utils.make_grid((fake + 1) / 2), global_step=epoch)


# TODO: reuse vars


def build_graph(config):
    # inputs / outputs #################################################################################################

    inputs = {
        'real': tf.compat.v1.placeholder(tf.float32, [None, 3, config.image_size, config.image_size])
    }
    outputs = {
        'd': {},
        'g': {},
        'fake': generator(tf.random.normal((config.batch_size, config.latent_size)))
    }

    # discriminator ####################################################################################################

    # d / real
    scores = discriminator(inputs['real'])
    loss_real = tf.nn.softplus(-scores)

    # d / fake
    scores = discriminator(outputs['fake'])
    loss_fake = tf.nn.softplus(scores)

    # d / opt
    outputs['d']['loss'] = loss_real + loss_fake
    # FIXME: only vars for a model
    outputs['d']['update'] = tf.optimizers.Adam(
        learning_rate=config.opt.lr,
        beta_1=config.opt.beta[0],
        beta_2=config.opt.beta[1],
        epsilon=1e-8,
    ).minimize(tf.reduce_mean(outputs['d']['loss']))

    # generator ########################################################################################################

    # g / fake
    scores = discriminator(outputs['fake'])
    loss_fake = F.softplus(-scores)

    # g / opt
    outputs['g']['loss'] = loss_fake
    # FIXME: only vars for a model
    outputs['g']['update'] = tf.optimizers.Adam(
        learning_rate=config.opt.lr,
        beta_1=config.opt.beta[0],
        beta_2=config.opt.beta[1],
        epsilon=1e-8,
    ).minimize(tf.reduce_mean(outputs['g']['loss']))

    return inputs, outputs


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


if __name__ == '__main__':
    main()
