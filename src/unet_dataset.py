from __future__ import division, print_function

import glob
import os

import numpy as np

import torch
from PIL import Image
from torch.utils.data import Dataset


class SatelliteImageDataset(Dataset):
    """xView2 dataset.

    Parameters
    ----------
    images_dir: str
        Directory with all the images.
    targets_dir: str
        Directory with all the targets.
    transform: callable, optional
        Optional transform to be applied on a sample. Default is `None`.
    """

    def __init__(self, images_dir, targets_dir, transform=None):
        self.images_list = sorted(glob.glob(images_dir + '/*.png'))
        self.targets_list = sorted(glob.glob(targets_dir + '/*.png'))

        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.images_list[idx])
        target = Image.open(self.targets_list[idx])

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target
