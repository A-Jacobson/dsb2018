import os
from glob import glob

from PIL import Image
from torch.utils.data import Dataset

from utils.segmentation import combine_masks
from utils.object_detection import extract_bboxes


class DSBDataset(Dataset):
    def __init__(self, root, transforms=None, merge_masks=True):
        self.root = root
        self.samples = glob(os.path.join(self.root, '*'))
        self.merge_masks = merge_masks
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = glob(os.path.join(self.samples[index], 'images', '*'))[0]
        masks_path = glob(os.path.join(self.samples[index], 'masks', '*'))
        image = Image.open(image_path).convert('RGB')
        mask = [Image.open(mask).convert('L') for mask in masks_path]

        if self.merge_masks:
            mask = combine_masks(mask)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        if not self.merge_masks:
            boxes = extract_bboxes(mask)
            return image, mask, boxes  # (x1, y1, x2, y2) boxes

        return image, mask

    def __len__(self):
        return len(self.samples)
