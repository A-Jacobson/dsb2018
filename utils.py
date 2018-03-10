import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.morphology import label
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import pad_to_factor
import hyperparams as hp


def combine_masks(masks):
    """
    :param masks: list of PIL images
    :return: combined PIL masks
    """
    combined_mask = np.zeros(np.array(masks[0]).shape, dtype='uint8')
    for mask in masks:
        combined_mask[np.array(mask) > 0] = 255
    return Image.fromarray(combined_mask)


def rle_encode(mask_image):
    pixels = mask_image.T.flatten()
    pixels = np.pad(pixels, pad_width=(1, 1), mode='constant')
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def rle_decode(rle, shape):
    '''
    rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def mask_to_rles(mask, threshold=0.5):
    """
    splits combined binary mask into encoder instances by clustering connected regions
    Returns per instance run length encodings.
    """
    lab_img = label(mask > threshold)  # Label connected regions of an integer array.
    for i in range(1, lab_img.max() + 1):
        yield rle_encode(lab_img == i)


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
        loss = F.binary_cross_entropy_with_logits(output, pad_to_factor(mask))
        val_loss += loss.data[0]
        if batch == random_batch:
            writer.add_image('image', inversenormalize(image.data.cpu()), step)
            writer.add_image('output', F.sigmoid(output.data), step)
            writer.add_image('target', mask.data, step)
    val_loss /= len(loader)
    writer.add_scalar('val loss', val_loss, step)
    return val_loss


def save(model, step, val_loss, exp_name, path):
    checkpoint = dict(state=model.state_dict(),
                      step=step,
                      val_loss=val_loss,
                      exp_name=exp_name)
    torch.save(checkpoint, path)


def get_mean_std_of_dataset(dataset):
    total_mean = np.array([0., 0., 0.])
    total_std = np.array([0., 0., 0.])
    for sample in dataset:
        img, _ = sample
        mean = np.mean(img.numpy(), axis=(1, 2))
        std = np.std(img.numpy(), axis=(1, 2))
        total_mean += mean
        total_std += std
    avg_mean = total_mean / len(dataset)
    avg_std = total_std / len(dataset)
    print(f'mean: {avg_mean}', f'stdev: {avg_std}')
    return avg_mean, avg_std


def inversenormalize(img, mean=hp.mean, std=hp.std):
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        return img * std.view(3, 1, 1) + mean.view(3, 1, 1)
