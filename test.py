import numpy as np
import torch

import matplotlib.pyplot as plt
import torchvision as tv
from torch.utils.data import DataLoader

from src.unet_dataset import SatelliteImageDataset

transforms = tv.transforms.Compose([
    tv.transforms.FiveCrop(512),
    tv.transforms.Lambda(lambda crops: torch.stack(
        [tv.transforms.ToTensor()(crop) for crop in crops]))
])

# Instantiate the new Dataset class to test it
satellite_image_dataset = SatelliteImageDataset(images_dir='data/images/',
                                                targets_dir='data/targets/',
                                                transform=transforms)

# The Dataloader
# Features include Batching, Shuffling and parallel loading
dataloader = DataLoader(satellite_image_dataset,
                        batch_size=4, shuffle=True, num_workers=2)

img, target = next(iter(dataloader))
img = img.view(-1, 3, 512, 512)
target = target.view(-1, 1, 512, 512)

plt.imshow(img[0].numpy().transpose(1, 2, 0), cmap='gray')
plt.imshow(target[0].squeeze().numpy(), cmap='jet', alpha=0.3)

plt.show()
