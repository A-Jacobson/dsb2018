import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms as trans


class JointCompose:
    def __init__(self, transforms, instance_masks=False):
        self.transforms = transforms
        if instance_masks:
            for t in self.transforms:
                t.instance_masks = instance_masks

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop:
    def __init__(self, height, width, instance_masks=False):
        self.height = height
        self.width = width
        self.instance_masks = instance_masks

    def __call__(self, img, mask):
        x, y, h, w = trans.RandomCrop.get_params(img, (self.height, self.width))
        img = F.crop(img, x, y, h, w)
        if self.instance_masks:
            mask = [F.crop(m, x, y, h, w) for m in mask]
        else:
            mask = F.crop(mask, x, y, h, w)
        return img, mask


class RandomRotate:
    def __init__(self, min_angle, max_angle, instance_masks=False):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.instance_masks = instance_masks

    def __call__(self, img, mask):
        degrees = trans.RandomRotation.get_params(degrees=(self.min_angle,
                                                           self.max_angle))
        img = F.rotate(img, degrees)
        if self.instance_masks:
            mask = [F.rotate(m, degrees) for m in mask]
        else:
            mask = F.rotate(mask, degrees)
        return img, mask


class Resize:
    def __init__(self, size, instance_masks=False, interpolation=Image.NEAREST):
        self.size = size
        self.instance_masks = instance_masks
        self.interpolation = interpolation

    def __call__(self, img, mask):
        img = F.resize(img, self.size, interpolation=self.interpolation)
        if self.instance_masks:
            mask = [F.resize(m, self.size, interpolation=self.interpolation) for m in mask]
        else:
            mask = F.resize(mask, self.size, interpolation=self.interpolation)
        return img, mask


class ToTensor:
    def __init__(self, instance_masks=False):
        self.instance_masks = instance_masks

    def __call__(self, img, mask):
        img_tensor = F.to_tensor(img)
        if self.instance_masks:
            mask_tensor = torch.cat([F.to_tensor(m) for m in mask], dim=0)
        else:
            mask_tensor = F.to_tensor(mask)
        return img_tensor, mask_tensor


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img = F.normalize(img, self.mean, self.std)
        return img, mask