import numpy as np
import torch
from skimage.morphology import label

import hyperparams as hp


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


def save(model, step, val_loss, exp_name, path):
    checkpoint = dict(state=model.state_dict(),
                      step=step,
                      val_loss=val_loss,
                      exp_name=exp_name)
    torch.save(checkpoint, path)


def get_mean_std(dataset):
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


def inverse_normalize(img, mean=hp.mean, std=hp.std):
    mean = torch.Tensor(mean)
    std = torch.Tensor(std)
    return img * std.view(3, 1, 1) + mean.view(3, 1, 1)

