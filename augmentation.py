"""Model Trainer

author: Masahiro Hayashi

This script defines custom image transformations that simultaneously transform
both images and segmentation masks.
"""
import torchvision.transforms.functional as TF
# from torchvision.transforms import Compose
from torchvision import transforms
import torch
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class GaussianNoise:
    """Apply Gaussian noise to tensor."""

    def __init__(self, mean=0., std=1., p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        noise = 0
        if random.random() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class DoubleToTensor:
    """Apply horizontal flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight=None):
        if weight is None:
            return TF.to_tensor(image), TF.to_tensor(mask)
        weight = weight.view(1, *weight.shape)
        return TF.to_tensor(image), TF.to_tensor(mask), weight

    def __repr__(self):
        return self.__class__.__name__ + '()'

class DoubleHorizontalFlip:
    """Apply horizontal flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight=None):
        p = random.random()
        if p < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if weight is None:
            return image, mask
        elif p > self.p:
            weight = TF.hflip(weight)
        return image, mask, weight

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class DoubleVerticalFlip:
    """Apply vertical flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight=None):
        p = random.random()
        if p < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if weight is None:
            return image, mask
        elif p > self.p:
            weight = TF.hflip(weight)
        return image, mask, weight

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class DoubleElasticTransform:
    """Based on implimentation on
    https://gist.github.com/erniejunior/601cdf56d2b424757de5"""

    def __init__(self, alpha=250, sigma=10, p=0.5, seed=None, randinit=True):
        if not seed:
            seed = random.randint(1, 100)
        self.random_state = np.random.RandomState(seed)
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.randinit = randinit


    def __call__(self, image, mask, weight=None):
        if random.random() < self.p:
            if self.randinit:
                seed = random.randint(1, 100)
                self.random_state = np.random.RandomState(seed)
                self.alpha = random.uniform(100, 300)
                self.sigma = random.uniform(10, 15)
                # print(self.alpha)
                # print(self.sigma)

            dim = image.shape
            dx = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            dy = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            image = image.view(*dim[1:]).numpy()
            mask = mask.view(*dim[1:]).numpy()
            x, y = np.meshgrid(np.arange(dim[1]), np.arange(dim[2]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            image = map_coordinates(image, indices, order=1)
            mask = map_coordinates(mask, indices, order=1)
            image, mask = image.reshape(dim), mask.reshape(dim)
            image, mask = torch.Tensor(image), torch.Tensor(mask)
            if weight is None:
                return image, mask
            weight = weight.view(*dim[1:]).numpy()
            weight = map_coordinates(weight, indices, order=1)
            weight = weight.reshape(dim)
            weight = torch.Tensor(weight)

        return (image, mask) if weight is None else (image, mask, weight)


class DoubleCompose(transforms.Compose):

    def __call__(self, image, mask, weight=None):
        if weight is None:
            for t in self.transforms:
                image, mask = t(image, mask)
            return image, mask
        for t in self.transforms:
            image, mask, weight = t(image, mask, weight)
        return image, mask, weight

###############################################################################
# For testing
###############################################################################
if __name__ == '__main__':
    # import torch
    X = np.random.rand(512, 512, 1)
    y = np.random.rand(512, 512, 1)
    # X, y = DoubleToTensor()(X, y)
    # X, y = DoubleVerticalFlip()(X, y)
    # X, y = DoubleHorizontalFlip()(X, y)
    import os
    from skimage import io
    from matplotlib import pyplot as plt
    image_path = os.getcwd() + '/data/train-volume.tif'
    mask_path = os.getcwd() + '/data/train-labels.tif'
    images = io.imread(image_path)
    masks = io.imread(mask_path)

    image = images[0]
    mask = masks[0]

    mean = 0.495
    std = 0.173
    out_size = 388

    image_mask_transform = DoubleCompose([
        DoubleToTensor(),
        DoubleElasticTransform(alpha=250, sigma=10),
        DoubleHorizontalFlip(),
        DoubleVerticalFlip(),
    ])

    # image_transform = transforms.Compose([
    #     transforms.Normalize(mean, std),
    #     transforms.Pad(30, padding_mode='reflect')
    # ])

    # mask_transform = transforms.CenterCrop(388)

    image_t, mask_t = image_mask_transform(image, mask)
    image_t, mask_t = image_t.numpy()[0], mask_t.numpy()[0]
    print(image_t.shape)
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(image)
    ax.set_title('Image')
    ax = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(image_t)
    ax.set_title('Transformed Image')
    ax = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(mask)
    ax.set_title('Label')
    ax = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(mask_t)
    ax.set_title('Transformed Label')
    fig.tight_layout()

    plt.show()
