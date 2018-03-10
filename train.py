import argparse

import hyperparams as hp
from models import Unet
from datasets import DSBDataset
from torch.optim import Adam
import torch

from utils import train
import transforms as segtrans
from sgdr import LRFinderScheduler, SGDRScheduler


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

    val_transforms = segtrans.JointCompose([segtrans.ToTensor(),
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



