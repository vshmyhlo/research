import os

import click
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.config import load_config
from all_the_tools.metrics import Last, Mean
from all_the_tools.torch.utils import Saver
from tensorboardX import SummaryWriter
from tqdm import tqdm

from tacotron.dataset import LJ
from tacotron.model.__init__ import Model
from tacotron.sampler import BatchSampler
from tacotron.utils import collate_fn
from tacotron.utils import compute_sample_sizes
from tacotron.utils import griffin_lim
from tacotron.vocab import CharVocab
from transforms import ApplyTo, ToTorch, Extract
from transforms.audio import LoadAudio
from transforms.text import VocabEncode, Normalize
from utils import WarmupCosineAnnealingLR
from utils import compute_nrow
from utils import entropy

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

    vocab = CharVocab()
    train_transform, eval_transform = build_transforms(vocab, config)

    train_dataset = LJ(config.dataset_path, subset='train', transform=train_transform)
    eval_dataset = LJ(config.dataset_path, subset='test', transform=eval_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=BatchSampler(
            compute_sample_sizes(train_dataset), batch_size=config.train.batch_size, shuffle=True, drop_last=True),
        num_workers=config.workers,
        collate_fn=collate_fn)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_sampler=BatchSampler(
            compute_sample_sizes(eval_dataset), batch_size=config.eval.batch_size, shuffle=False, drop_last=False),
        num_workers=config.workers,
        collate_fn=collate_fn)

    mean_std = torch.load('./tacotron/spectrogram_stats.pth')
    model = Model(config.model, vocab_size=len(vocab), sample_rate=config.sample_rate, mean_std=mean_std).to(DEVICE)
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = build_scheduler(optimizer, config, len(train_data_loader))
    saver = Saver({
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
    })
    if config.restore_path is not None:
        saver.load(config.restore_path, keys=['model'])

    for epoch in range(1, config.train.epochs + 1):
        train_epoch(model, train_data_loader, optimizer, scheduler, epoch=epoch, config=config)
        eval_epoch(model, eval_data_loader, epoch=epoch, config=config)
        saver.save(
            os.path.join(config.experiment_path, 'checkpoint_{}.pth'.format(epoch)),
            epoch=epoch)


def build_optimizer(parameters, config):
    if config.train.opt.type == 'adam':
        optimizer = torch.optim.Adam(
            parameters,
            config.train.opt.lr,
            betas=config.train.opt.beta,
            eps=config.train.opt.eps,
            weight_decay=config.train.opt.weight_decay)
    else:
        raise AssertionError('invalid optimizer {}'.format(config.train.opt.type))

    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    if config.train.sched.type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            [epoch * steps_per_epoch for epoch in config.train.sched.epochs],
            gamma=0.1)
    elif config.train.sched.type == 'warmup_cosine':
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            epoch_warmup=config.train.sched.epochs_warmup * steps_per_epoch,
            epoch_max=config.train.epochs * steps_per_epoch)
    else:
        raise AssertionError('invalid scheduler {}'.format(config.train.sched.type))

    return scheduler


def build_transforms(vocab, config):
    train_transform = eval_transform = T.Compose([
        ApplyTo('text', T.Compose([
            Normalize(),
            VocabEncode(vocab),
            ToTorch(),
        ])),
        ApplyTo('audio', T.Compose([
            LoadAudio(config.sample_rate),
            ToTorch(),
        ])),
        Extract(['text', 'audio'])
    ])

    return train_transform, eval_transform


def train_epoch(model, data_loader, optimizer, scheduler, epoch, config):
    metrics = {
        'loss': Mean(),
        'lr': Last(),
    }

    model.train()
    for (text, text_mask), (audio, audio_mask) in \
            tqdm(data_loader, desc='epoch {}/{}, train'.format(epoch, config.train.epochs)):
        text, audio, text_mask, audio_mask = \
            [x.to(DEVICE) for x in [text, audio, text_mask, audio_mask]]

        output, pre_output, target, target_mask, weight = model(text, text_mask, audio, audio_mask)

        loss = masked_mse(output, target, target_mask) + \
               masked_mse(pre_output, target, target_mask)

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_last_lr()))

        optimizer.zero_grad()
        loss.mean().backward()
        if config.train.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
        optimizer.step()
        scheduler.step()

    writer = SummaryWriter(os.path.join(config.experiment_path, 'train'))
    with torch.no_grad():
        gl_true = griffin_lim(target, model.spectra)
        gl_pred = griffin_lim(output, model.spectra)
        output, pre_output, target, weight = \
            [x.unsqueeze(1) for x in [output, pre_output, target, weight]]
        nrow = compute_nrow(target)

        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image('target', torchvision.utils.make_grid(
            target, nrow=nrow, normalize=True), global_step=epoch)
        writer.add_image('output', torchvision.utils.make_grid(
            output, nrow=nrow, normalize=True), global_step=epoch)
        writer.add_image('pre_output', torchvision.utils.make_grid(
            pre_output, nrow=nrow, normalize=True), global_step=epoch)
        writer.add_image('weight', torchvision.utils.make_grid(
            weight, nrow=nrow, normalize=True), global_step=epoch)
        for i in tqdm(range(min(text.size(0), 4)), desc='writing audio'):
            writer.add_audio(
                'audio/{}'.format(i), audio[i], sample_rate=config.sample_rate, global_step=epoch)
            writer.add_audio(
                'griffin-lim-true/{}'.format(i), gl_true[i], sample_rate=config.sample_rate, global_step=epoch)
            writer.add_audio(
                'griffin-lim-pred/{}'.format(i), gl_pred[i], sample_rate=config.sample_rate, global_step=epoch)

    writer.flush()
    writer.close()


def eval_epoch(model, data_loader, epoch, config):
    metrics = {
        'loss': Mean(),
    }

    with torch.no_grad():
        model.eval()
        for (text, text_mask), (audio, audio_mask) in \
                tqdm(data_loader, desc='epoch {}/{}, eval'.format(epoch, config.train.epochs)):
            text, audio, text_mask, audio_mask = \
                [x.to(DEVICE) for x in [text, audio, text_mask, audio_mask]]

            output, pre_output, target, target_mask, weight = model(text, text_mask, audio, audio_mask)

            loss = masked_mse(output, target, target_mask) + \
                   masked_mse(pre_output, target, target_mask)

            metrics['loss'].update(loss.data.cpu().numpy())

    writer = SummaryWriter(os.path.join(config.experiment_path, 'eval'))
    with torch.no_grad():
        gl_true = griffin_lim(target, model.spectra)
        gl_pred = griffin_lim(output, model.spectra)
        output, pre_output, target, weight = \
            [x.unsqueeze(1) for x in [output, pre_output, target, weight]]
        nrow = compute_nrow(target)

        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image('target', torchvision.utils.make_grid(
            target, nrow=nrow, normalize=True), global_step=epoch)
        writer.add_image('output', torchvision.utils.make_grid(
            output, nrow=nrow, normalize=True), global_step=epoch)
        writer.add_image('pre_output', torchvision.utils.make_grid(
            pre_output, nrow=nrow, normalize=True), global_step=epoch)
        writer.add_image('weight', torchvision.utils.make_grid(
            weight, nrow=nrow, normalize=True), global_step=epoch)
        for i in tqdm(range(min(text.size(0), 4)), desc='writing audio'):
            writer.add_audio(
                'audio/{}'.format(i), audio[i], sample_rate=config.sample_rate, global_step=epoch)
            writer.add_audio(
                'griffin-lim-true/{}'.format(i), gl_true[i], sample_rate=config.sample_rate, global_step=epoch)
            writer.add_audio(
                'griffin-lim-pred/{}'.format(i), gl_pred[i], sample_rate=config.sample_rate, global_step=epoch)

    writer.flush()
    writer.close()


def masked_mse(input, target, mask):
    loss = (input - target)**2
    loss = loss.sum(1)  # sum by C
    loss = (loss * mask).sum(1) / mask.sum(1)  # mean by T

    return loss


def attention_entropy_loss(input, mask):
    per_ent = entropy(input, dim=1)
    per_ent = (per_ent * mask).sum(1) / mask.sum(1)  # mean by T

    mask = mask.unsqueeze(1)
    mean_ent = (input * mask).sum(2) / mask.sum(2)  # mean by T
    mean_ent = entropy(mean_ent)

    return per_ent - mean_ent


if __name__ == '__main__':
    main()
