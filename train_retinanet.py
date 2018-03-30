import numpy as np
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import transforms as segtrans
from criterion import FocalLoss
from datasets import DSBDataset
from models import RetinaNet
from sgdr import SGDRScheduler
from utils.data import save
from utils.object_detection import AnchorHelper


def train(model, optimizer, scheduler, focal_loss, train_dataset, val_dataset=None,
          n_epochs=10, batch_size=32, step=0, exp_name=None, device=1):
    step = step
    best_loss = float('inf')
    model.train()
    writer = SummaryWriter(f'runs/{exp_name}')
    loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=4, shuffle=True)
    for epoch in tqdm(range(n_epochs), total=n_epochs):
        pbar = tqdm(loader, total=len(loader))
        for images, class_labels, anchor_deltas in pbar:
            images = Variable(images).cuda()
            class_labels = Variable(class_labels).cuda()
            anchor_deltas = Variable(anchor_deltas).cuda()
            class_preds, box_preds = model(images)
            class_loss = focal_loss(class_preds, class_labels)
            box_loss = F.smooth_l1_loss(box_preds, anchor_deltas)
            loss = class_loss + box_loss
            optimizer.zero_grad()
            loss.backward()
            scheduler.step()
            optimizer.step()
            pbar.set_description(f'[Loss: {loss.data[0]:.4f}]')
            writer.add_scalar('class_loss', class_loss.data[0], step)
            writer.add_scalar('box_loss', box_loss.data[0], step)
            writer.add_scalar('loss', loss.data[0], step)
            writer.add_scalar('lr', scheduler.lr, step)
            step += 1
        val_loss = eval(model, focal_loss, val_dataset, batch_size, writer, step, device) if val_dataset else float(
            'inf')
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), step, bins='doane')
        if val_loss <= best_loss:
            best_loss = val_loss
            save(model, step, val_loss, exp_name, f'checkpoints/{exp_name}.pt')


def eval(model, focal_loss, val_dataset, batch_size, writer, step, device):
    model.eval()
    loader = DataLoader(val_dataset, batch_size=1, pin_memory=True, num_workers=4, shuffle=False)
    val_loss = 0
    val_class_loss = 0
    val_box_loss = 0
    random_batch = np.random.randint(0, len(loader))
    for batch, (images, class_labels, anchor_deltas) in enumerate(loader):
        images = Variable(images).cuda()
        class_labels = Variable(class_labels).cuda()
        anchor_deltas = Variable(anchor_deltas).cuda()
        class_preds, box_preds = model(images)
        class_loss = focal_loss(class_preds, class_labels)
        box_loss = F.smooth_l1_loss(box_preds, anchor_deltas)
        val_class_loss += class_loss.data[0]
        val_box_loss += box_loss.data[0]
        val_loss += (class_loss + box_loss).data[0]
        # if batch == random_batch:
        #     writer.add_image('image', image.data.cpu(), step)
        #     writer.add_image('output', F.sigmoid(output.data), step)
        #     writer.add_image('target', mask.data, step)
    val_loss /= len(loader)
    writer.add_scalar('val loss', val_loss, step)
    writer.add_scalar('val loss class', val_class_loss, step)
    writer.add_scalar('val loss box', val_box_loss, step)

    return val_loss


def main():
    ROOT = '/home/austin/data/dsb/train'

    transforms = segtrans.JointCompose([segtrans.Resize(300),
                                        segtrans.RandomCrop(256, 256),
                                        segtrans.ToTensor()], instance_masks=True)

    anchor_helper = AnchorHelper(areas=(16, 32, 64, 128, 256),
                                 positive_overlap=0.5,
                                 negative_overlap=0.4)

    dataset = DSBDataset(ROOT, transforms, merge_masks=False, anchor_helper=anchor_helper)

    model = RetinaNet(num_classes=2)
    model.cuda()
    optimizer = Adam(model.parameters(), lr=1e-5)
    focal_loss = FocalLoss(gamma=2, alpha=1e3, ignore_index=-1)
    scheduler = SGDRScheduler(optimizer, min_lr=1e-7,
                              max_lr=1e-6, cycle_length=400, current_step=0)
    train(model, optimizer, scheduler, focal_loss, dataset,
          n_epochs=20, batch_size=12,
          exp_name='retinacat')


if __name__ == '__main__':
    main()
