import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import hyperparams as hp
import transforms as segtrans
from datasets import DSBDataset
from models import Unet
from sgdr import LRFinderScheduler, SGDRScheduler
from utils.data import save, inverse_normalize

parser = argparse.ArgumentParser(description='DSB Nuclei Training')
parser.add_argument('--data',
                    default=hp.root,
                    help='path to dataset')
parser.add_argument('--epochs', '-e', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', '-bs', default=6, type=int, help='mini-batch size (default: 12)')
parser.add_argument('--name', '-n', default='unet', help='experiment name', type=str)
parser.add_argument('--find_lr', default=False, type=bool,
                    help='runs training with LR finding scheduler,'
                         ' check tensorboard plots to choose max_lr')
parser.add_argument('--checkpoint', '-cp', default=None, type=str, help='path to checkpoint')
parser.add_argument('--device', default=1, type=int, help='which gpu to use')


def train(model, optimizer, scheduler, train_dataset, val_dataset,
          n_epochs=10, batch_size=32, step=0, exp_name=None, device=1):
    step = step
    best_loss = float('inf')
    model.train()
    writer = SummaryWriter(f'runs/{exp_name}')
    loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=4, shuffle=True)
    for epoch in tqdm(range(n_epochs), total=n_epochs):
        pbar = tqdm(loader, total=len(loader))
        for image, mask in pbar:
            image, mask = Variable(image).cuda(device), Variable(mask).cuda(device)
            output = model(image)
            loss = F.binary_cross_entropy_with_logits(output, mask)
            optimizer.zero_grad()
            loss.backward()
            scheduler.step()
            optimizer.step()
            pbar.set_description(f'[Loss: {loss.data[0]:.4f}]')
            writer.add_scalar('loss', loss.data[0], step)
            writer.add_scalar('lr', scheduler.lr, step)
            step += 1
        val_loss = eval(model, val_dataset, batch_size, writer, step, device)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), step, bins='doane')
        if val_loss < best_loss:
            best_loss = val_loss
            save(model, step, val_loss, exp_name, f'checkpoints/{exp_name}.pt')


def eval(model, val_dataset, batch_size, writer, step, device):
    model.eval()
    loader = DataLoader(val_dataset, batch_size=1, pin_memory=True, num_workers=4, shuffle=False)
    val_loss = 0
    random_batch = np.random.randint(0, len(loader))
    for batch, (image, mask) in enumerate(loader):
        image, mask = Variable(image, volatile=True).cuda(device=device), Variable(mask, volatile=True).cuda(device)
        output = model(image)
        loss = F.binary_cross_entropy_with_logits(output, mask)
        val_loss += loss.data[0]
        if batch == random_batch:
            writer.add_image('image', inverse_normalize(image.data.cpu()), step)
            writer.add_image('output', F.sigmoid(output.data), step)
            writer.add_image('target', mask.data, step)
    val_loss /= len(loader)
    writer.add_scalar('val loss', val_loss, step)
    return val_loss


def main():
    args = parser.parse_args()
    step = 0
    exp_name = f'{args.name}_{hp.max_lr}_{hp.cycle_length}'

    transforms = segtrans.JointCompose([segtrans.Resize(400),
                                        segtrans.RandomRotate(0, 90),
                                        segtrans.RandomCrop(256, 256),
                                        segtrans.ToTensor(),
                                        segtrans.Normalize(mean=hp.mean,
                                                           std=hp.std)])

    val_transforms = segtrans.JointCompose([segtrans.PadToFactor(),
                                            segtrans.ToTensor(),
                                            segtrans.Normalize(mean=hp.mean,
                                                               std=hp.std)])

    train_dataset = DSBDataset(f'{args.data}/train', transforms=transforms)
    val_dataset = DSBDataset(f'{args.data}/val', transforms=val_transforms)

    model = Unet()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state'])
        step = checkpoint['step']
        exp_name = checkpoint['exp_name']

    optimizer = Adam(model.parameters(), lr=hp.max_lr)

    if args.find_lr:
        scheduler = LRFinderScheduler(optimizer)
    else:
        scheduler = SGDRScheduler(optimizer, min_lr=hp.min_lr,
                                  max_lr=hp.max_lr, cycle_length=hp.cycle_length, current_step=step)

    model.cuda(device=args.device)
    train(model, optimizer, scheduler, train_dataset, val_dataset,
          n_epochs=args.epochs, batch_size=args.batch_size,
          exp_name=exp_name, device=args.device, step=step)


if __name__ == '__main__':
    main()
