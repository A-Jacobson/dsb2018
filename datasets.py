import os
from glob import glob

from PIL import Image
from torch.utils.data import Dataset

from utils import combine_masks


class DSBDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.samples = glob(os.path.join(self.root, '*'))
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = glob(os.path.join(self.samples[index], 'images', '*'))[0]
        masks_path = glob(os.path.join(self.samples[index], 'masks', '*'))
        image = Image.open(image_path).convert('RGB')
        masks = [Image.open(mask).convert('L') for mask in masks_path]
        mask = combine_masks(masks)
        if self.transforms:
            image, mask = self.transforms(image, mask)
        return image, mask

    def __len__(self):
        return len(self.samples)
