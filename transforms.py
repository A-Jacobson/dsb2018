import torch
from torchvision import transforms as trans
import torchvision.transforms.functional as F
from PIL import Image


class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, mask):
        i, j, h, w = trans.RandomCrop.get_params(img, (self.height, self.width))
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return img, mask


class RandomRotate:
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, img, mask):
        degrees = trans.RandomRotation.get_params(degrees=(self.min_angle,
                                                            self.max_angle))
        img = F.rotate(img, degrees)
        mask = F.rotate(mask, degrees)
        return img, mask


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        img = F.resize(img, self.size, interpolation=Image.NEAREST)
        mask = F.resize(mask, self.size, interpolation=Image.NEAREST)
        return img, mask


class ToTensor:

    def __call__(self, img, mask):
        return F.to_tensor(img), F.to_tensor(mask)


class Normalize:
    """
    Normalizes img, leaves mask untouched.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img = F.normalize(img, self.mean, self.std)
        return img, mask



