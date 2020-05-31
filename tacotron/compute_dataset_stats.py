import os

import click
import torch
import torch.utils.data
from all_the_tools.config import load_config
from tqdm import tqdm

from tacotron.dataset import LJ
from tacotron.model import Model
from tacotron.sampler import BatchSampler
from tacotron.train import build_transforms
from tacotron.utils import collate_fn, compute_sample_sizes
from tacotron.vocab import CharVocab

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--dataset-path', type=click.Path(), required=True)
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(config_path, **kwargs):
    config = load_config(
        config_path,
        **kwargs)

    vocab = CharVocab()
    train_transform, eval_transform = build_transforms(vocab, config)

    train_dataset = LJ(config.dataset_path, subset='train', transform=train_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=BatchSampler(
            compute_sample_sizes(train_dataset), batch_size=config.train.batch_size, shuffle=True, drop_last=True),
        num_workers=config.workers,
        collate_fn=collate_fn)

    mean_std = torch.zeros(config.model.num_mels), torch.ones(config.model.num_mels)
    model = Model(config.model, vocab_size=len(vocab), sample_rate=config.sample_rate, mean_std=mean_std).to(DEVICE)

    compute_stats(model, train_data_loader)


def compute_stats(model, data_loader):
    with torch.no_grad():
        model.eval()
        all_targets = []
        for (text, text_mask), (audio, audio_mask) in \
                tqdm(data_loader):
            text, audio, text_mask, audio_mask = \
                [x.to(DEVICE) for x in [text, audio, text_mask, audio_mask]]

            output, pre_output, target, target_mask, weight = model(text, text_mask, audio, audio_mask)
            for t, t_m in zip(target, target_mask):
                all_targets.append(t[:, t_m].cpu())

        all_targets = torch.cat(all_targets, 1)
        torch.save((all_targets.mean(1), all_targets.std(1)), './tacotron/spectrogram_stats.pth')


if __name__ == '__main__':
    main()
