import argparse
import utils
import torch.nn.functional as F
import os
from ticpfptp.metrics import Mean
from dataset import Dataset
import torch.utils.data
import torch
import logging
from tqdm import tqdm
from ticpfptp.format import args_to_string
from ticpfptp.torch import fix_seed
from vae.model import Encoder, Decoder
from tensorboardX import SummaryWriter


# TODO: spherical z
# TODO: spherical interpolation
# TODO: norm z
# TODO: cleanup (args, code)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, default='./tf_log')
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--dataset-path', type=str, default='./data')
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--model-size', type=int, default=32)
    parser.add_argument('--latent-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    # parser.add_argument('--opt', type=str, choices=['adam', 'momentum'], default='momentum')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)

    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    logging.info(args_to_string(args))
    fix_seed(args.seed)

    data_loader = torch.utils.data.DataLoader(
        Dataset(args.dataset_path),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(args.model_size, args.latent_size)
    decoder = Decoder(args.model_size, args.latent_size)
    encoder.to(device)
    decoder.to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)
    noise_dist = torch.distributions.Normal(0, 1)

    writer = SummaryWriter(args.experiment_path)
    metrics = {
        'loss': Mean()
    }

    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        for real, _ in tqdm(data_loader, desc='epoch {} training'.format(epoch)):
            real = real.to(device)

            mean, log_var = encoder(real)
            latent = noise_dist.sample((args.batch_size, args.latent_size)).to(device)
            latent = mean + latent * torch.exp(log_var / 2)
            fake = decoder(latent)

            # TODO: loss (reconstruction, summing, mean)
            mse = F.mse_loss(input=fake, target=real)
            kld = -0.5 * (1 + log_var - mean**2 - log_var.exp()).sum(-1)
            loss = mse.mean() + kld.mean()
            metrics['loss'].update(loss.data.cpu().numpy())

            opt.zero_grad()
            loss.mean().backward()
            opt.step()

        writer.add_scalar('loss', metrics['loss'].compute_and_reset(), global_step=epoch)
        writer.add_image('real', utils.make_grid((real + 1) / 2), global_step=epoch)
        writer.add_image('fake', utils.make_grid((fake + 1) / 2), global_step=epoch)


if __name__ == '__main__':
    main()
