import os

import click
import higher
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.config import load_config
from all_the_tools.metrics import Mean, Last
from all_the_tools.torch.utils import Saver
from sklearn.model_selection import StratifiedKFold
from tensorboardX import SummaryWriter
from tqdm import tqdm

from mpl_torch.model import Model
from mpl_torch.utils import XUDataLoader
from scheduler import WarmupCosineAnnealingLR
from utils import compute_nrow, one_hot
from utils import cross_entropy, entropy

NUM_CLASSES = 10
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: pretrain


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

    x_transform, u_transform, eval_transform = build_transforms()
    x_indices, u_indices = build_x_u_split(
        torchvision.datasets.CIFAR10(config.dataset_path, train=True, download=True),
        config.train.num_labeled)

    x_dataset = torch.utils.data.Subset(
        torchvision.datasets.CIFAR10(config.dataset_path, train=True, transform=x_transform, download=True),
        x_indices)
    u_dataset = torch.utils.data.Subset(
        torchvision.datasets.CIFAR10(config.dataset_path, train=True, transform=u_transform, download=True),
        u_indices)
    eval_dataset = torchvision.datasets.CIFAR10(
        config.dataset_path, train=False, transform=eval_transform, download=True)

    train_data_loader = XUDataLoader(
        torch.utils.data.DataLoader(
            x_dataset,
            batch_size=config.train.x_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=config.workers),
        torch.utils.data.DataLoader(
            u_dataset,
            batch_size=config.train.u_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=config.workers))
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.eval.batch_size,
        num_workers=config.workers)

    model = nn.ModuleDict({
        'teacher': Model(NUM_CLASSES, config.train.teacher.dropout),
        'student': Model(NUM_CLASSES, config.train.student.dropout),
    }).to(DEVICE)
    model.apply(weights_init)

    opt_teacher = build_optimizer(model.teacher.parameters(), config.train.teacher)
    opt_student = build_optimizer(model.student.parameters(), config.train.student)

    sched_teacher = build_scheduler(opt_teacher, config, len(train_data_loader))
    sched_student = build_scheduler(opt_student, config, len(train_data_loader))

    saver = Saver({
        'model': model,
    })
    if config.restore_path is not None:
        saver.load(config.restore_path, keys=['model'])

    for epoch in range(1, config.epochs + 1):
        train_epoch(
            model,
            train_data_loader,
            opt_teacher=opt_teacher,
            opt_student=opt_student,
            sched_teacher=sched_teacher,
            sched_student=sched_student,
            epoch=epoch,
            config=config)
        if epoch % config.log_interval != 0:
            continue
        eval_epoch(
            model,
            eval_data_loader,
            epoch=epoch,
            config=config)
        saver.save(
            os.path.join(config.experiment_path, 'checkpoint_{}.pth'.format(epoch)),
            epoch=epoch)


# FIXME:
def build_x_u_split(dataset, num_labeled):
    targets = torch.tensor([target for _, target in tqdm(dataset, 'loading split')])

    ratio = len(dataset) // num_labeled

    u_indices, x_indices = next(StratifiedKFold(ratio, shuffle=True, random_state=42).split(targets, targets))
    u_indices, x_indices = torch.tensor(u_indices), torch.tensor(x_indices)

    return x_indices, u_indices


def build_optimizer(parameters, config):
    if config.opt.type == 'sgd':
        optimizer = torch.optim.SGD(
            parameters,
            config.opt.lr,
            momentum=config.opt.momentum,
            weight_decay=config.opt.weight_decay,
            nesterov=True)
    else:
        raise AssertionError('invalid optimizer {}'.format(config.opt.type))

    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    if config.train.sched.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            config.epochs * steps_per_epoch)
    elif config.train.sched.type == 'warmup_cosine':
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            epoch_warmup=int(config.epochs * steps_per_epoch * 0.1),
            epoch_max=config.epochs * steps_per_epoch)
    elif config.train.sched.type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=(config.epochs * steps_per_epoch) // 3,
            gamma=0.1)
    elif config.train.sched.type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            [epoch * steps_per_epoch for epoch in config.train.sched.epochs],
            gamma=0.1)
    else:
        raise AssertionError('invalid scheduler {}'.format(config.train.sched.type))

    return scheduler


def denormalize(input):
    input = input * 0.25 + 0.5

    return input


def build_transforms():
    to_tensor_and_norm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.25]),
    ])

    x_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        to_tensor_and_norm,
    ])
    u_transform = x_transform
    eval_transform = T.Compose([
        to_tensor_and_norm,
    ])

    return x_transform, u_transform, eval_transform


def train_epoch(model, data_loader, opt_teacher, opt_student, sched_teacher, sched_student, epoch, config):
    metrics = {
        'teacher/loss': Mean(),
        'teacher/grad_norm': Mean(),
        'teacher/lr': Last(),

        'student/loss': Mean(),
        'student/grad_norm': Mean(),
        'student/lr': Last(),
    }

    model.train()
    for (x_image, x_target), (u_image,) in \
            tqdm(data_loader, desc='epoch {}/{}, train'.format(epoch, config.epochs)):
        x_image, x_target, u_image = x_image.to(DEVICE), x_target.to(DEVICE), u_image.to(DEVICE)

        with higher.innerloop_ctx(model.student, opt_student) as (h_model_student, h_opt_student):
            # student ##################################################################################################

            loss_student = cross_entropy(input=h_model_student(u_image), target=model.teacher(u_image)).mean()
            metrics['student/loss'].update(loss_student.data.cpu().numpy())
            metrics['student/lr'].update(np.squeeze(sched_student.get_last_lr()))

            def grad_callback(grads):
                metrics['student/grad_norm'].update(
                    grad_norm(grads).data.cpu().numpy())

                return grads

            h_opt_student.step(loss_student.mean(), grad_callback=grad_callback)
            sched_student.step()

            # teacher ##################################################################################################

            loss_teacher = cross_entropy(input=model.teacher(x_image), target=one_hot(x_target, NUM_CLASSES)).mean() + \
                           cross_entropy(input=h_model_student(x_image), target=one_hot(x_target, NUM_CLASSES)).mean()

            metrics['teacher/loss'].update(loss_teacher.data.cpu().numpy())
            metrics['teacher/lr'].update(np.squeeze(sched_teacher.get_last_lr()))

            opt_teacher.zero_grad()
            loss_teacher.mean().backward()
            opt_teacher.step()
            metrics['teacher/grad_norm'].update(
                grad_norm(p.grad for p in model.teacher.parameters()).data.cpu().numpy())
            sched_teacher.step()

            # copy student weights #####################################################################################

            with torch.no_grad():
                for p, p_prime in zip(model.student.parameters(), h_model_student.parameters()):
                    p.copy_(p_prime)

    if epoch % config.log_interval != 0:
        return

    writer = SummaryWriter(os.path.join(config.experiment_path, 'train'))
    with torch.no_grad():
        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image('x_image', torchvision.utils.make_grid(
            denormalize(x_image), nrow=compute_nrow(x_image), normalize=True), global_step=epoch)
        writer.add_image('u_image', torchvision.utils.make_grid(
            denormalize(u_image), nrow=compute_nrow(u_image), normalize=True), global_step=epoch)

    writer.flush()
    writer.close()


def eval_epoch(model, data_loader, epoch, config):
    metrics = {
        'teacher/accuracy': Mean(),
        'teacher/entropy': Mean(),

        'student/accuracy': Mean(),
        'student/entropy': Mean(),
    }

    with torch.no_grad():
        model.eval()
        for x_image, x_target in tqdm(data_loader, desc='epoch {}/{}, eval'.format(epoch, config.epochs)):
            x_image, x_target = x_image.to(DEVICE), x_target.to(DEVICE)

            probs_teacher = model.teacher(x_image)
            probs_student = model.student(x_image)

            metrics['teacher/entropy'].update(entropy(probs_teacher).data.cpu().numpy())
            metrics['student/entropy'].update(entropy(probs_student).data.cpu().numpy())

            metrics['teacher/accuracy'].update(
                (probs_teacher.argmax(-1) == x_target).float().data.cpu().numpy())
            metrics['student/accuracy'].update(
                (probs_student.argmax(-1) == x_target).float().data.cpu().numpy())

    writer = SummaryWriter(os.path.join(config.experiment_path, 'eval'))
    with torch.no_grad():
        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image('x_image', torchvision.utils.make_grid(
            denormalize(x_image), nrow=compute_nrow(x_image), normalize=True), global_step=epoch)

    writer.flush()
    writer.close()


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear,)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d,)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def grad_norm(grads, norm_type=2):
    return torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type)


if __name__ == '__main__':
    main()


def tmp():
    teacher_optimizer = ...
    student_optimizer = ...

    with higher.innerloop_ctx(student, student_optimizer) as (smodel, sdiffopt):

        student.train()
        teacher.train()

        teacher_logits = teacher(unsupervised_batch)
        student_logits = smodel(unsupervised_batch)
        distillation_loss = soft_ce(
            torch.log_softmax(student_logits, dim=1),
            torch.softmax(teacher_logits, dim=1),
        )
        print("Distillation loss:", distillation_loss.item())

        sdiffopt.step(distillation_loss)

        student_logits = smodel(student_data)
        student_logits.squeeze_(dim=1)
        student_loss = ce(student_logits, student_labels)

        print("Student loss:", student_loss.item())

        student_loss.backward()
        print(
            "Teacher grad: {} +- {}".format(
                teacher.fc1.weight.grad.mean(), teacher.fc1.weight.grad.std()
            )
        )
        teacher_optimizer.step()

        if step % supervised_teacher_update_freq == 0:
            supervise_teacher(teacher_data, teacher_labels)

        clear_output(wait=True)
        with torch.no_grad():
            student.eval()
            teacher.eval()
            plot_predictions(
                (
                    unsupervised_data.numpy(),
                    student(unsupervised_data).numpy().argmax(1),
                    "Student",
                ),
                (
                    unsupervised_data.numpy(),
                    teacher(unsupervised_data).numpy().argmax(1),
                    "Teacher",
                ),
            )

        with torch.no_grad():
            for old_p, new_p in zip(student.parameters(), smodel.parameters()):
                old_p.copy_(new_p)
