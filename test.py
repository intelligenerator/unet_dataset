import random

import numpy as np

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as tf
from src.unet_dataset import SatelliteImageDataset
from torch.utils.data import DataLoader


def transforms(x, y):
    x = list(tf.five_crop(x, 512))

    y = list(tf.five_crop(y, 512))

    for i in range(len(x)):
        if random.random() > 0.5:
            x[i] = tf.hflip(x[i])
            y[i] = tf.hflip(y[i])

        if random.random() > 0.5:
            x[i] = tf.vflip(x[i])
            y[i] = tf.vflip(y[i])

        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            x[i] = tf.rotate(x[i], angle)
            y[i] = tf.rotate(y[i], angle)

    x = torch.stack([tf.to_tensor(crop) for crop in x])
    y = torch.stack([tf.to_tensor(crop) for crop in y])

    return x, y


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
