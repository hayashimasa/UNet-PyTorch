"""Cell Dataset

author: Masahiro Hayashi

This script defines the CellDataset object that preprocess the ISBI 2012
EM cell dataset and allows user to retrieve sample images using an iterator.
"""
import os
from tqdm import tqdm

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from skimage.segmentation import find_boundaries
import numpy as np
import matplotlib.pyplot as plt


from augmentation import (
    DoubleCompose, DoubleToTensor,
    DoubleHorizontalFlip, DoubleVerticalFlip, DoubleElasticTransform
)


class CellDataset(Dataset):
    """ISBI 2012 EM Cell dataset.
    """

    def __init__(
        self, root_dir=None,
        image_mask_transform=None, image_transform=None, mask_transform=None,
        pct=.9, data_type='train', in_size=572, out_size=388,
        w0=10, sigma=5, weight_map_dir=None
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            image_mask_transform (callable, optional): Optional
            transform to be applied on images and mask label simultaneuosly.
            image_transform (callable, optional): Optional
            transform to be applied on images.
            mask_transform (callable, optional): Optional
            transform to be applied on mask labels.
            pct (float): percentage of data to use as training data
            data_type (string): either 'train' or 'test'
            in_size (int): input size of image
            out_size (int): output size of segmentation map
        """
        self.root_dir = os.getcwd() if not root_dir else root_dir
        path = os.path.join(self.root_dir, 'data')
        self.train_path = os.path.join(path, 'train-volume.tif')
        self.mask_path = os.path.join(path, 'train-labels.tif')
        self.test_path = os.path.join(path, 'test-volume.tif')

        self.data_type = data_type
        self.image_mask_transform = image_mask_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.weight_transform = self.mask_transform
        if self.data_type == 'validate':
            self.weight_transform = transforms.Compose(
                self.mask_transform.transforms[1:]
            )
        self.n_classes = 2
        self.images = io.imread(self.train_path)
        self.masks = io.imread(self.mask_path)

        n = int(np.ceil(self.images.shape[0] * pct))

        if self.data_type == 'train':
            self.images = io.imread(self.train_path)[:n]
            self.masks = io.imread(self.mask_path)[:n]
        elif self.data_type == 'validate':
            self.images = io.imread(self.train_path)[n:]
            self.masks = io.imread(self.mask_path)[n:]

        self.mean = np.average(self.images)
        self.std = np.std(self.images)
        self.w0 = w0
        self.sigma = sigma
        # if weight_map_dir:
        #     self.weight_map = torch.load(weight_map_dir)
        #     print(self.weight_map)
        # if not weight_map_dir:
        self.weight_map = self._get_weights(self.w0, self.sigma)
        # torch.save(self.weight_map, 'weight_map.pt')

        self.in_size = in_size
        self.out_size = out_size
        # print(self.images.shape, 'images')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        """Returns a image sample from the dataset
        """
        image = self.images[idx]
        mask = self.masks[idx]
        weight = self.weight_map[idx]

        if self.image_mask_transform:
            image, mask, weight = self.image_mask_transform(
                image, mask, weight
            )
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            weight = self.weight_transform(mask)

        # img = Image.fromarray(255*label[0].numpy())
        # img.show()
        sample = {'image': image, 'mask': mask, 'weight': weight}

        return sample

    def _get_weights(self, w0, sigma):
        class_weight = self._get_class_weight(self.masks)
        # boundary_weight = self._get_boundary_weight(self.masks, w0, sigma)
        return class_weight  # + boundary_weight

    def _get_class_weight(self, target):
        n, H, W = target.shape
        weight = torch.zeros(n, H, W)
        for i in range(self.n_classes):
            i_t = i * torch.ones([n, H, W], dtype=torch.long)
            loc_i = (torch.Tensor(target // 255) == i_t).to(torch.long)
            count_i = loc_i.view(n, -1).sum(1)
            total = H * W
            weight_i = total / count_i
            weight_t = weight_i.view(-1, 1, 1) * loc_i
            weight += weight_t
        return weight

    def _get_boundary_weight(self, target, w0=10, sigma=5):
        """This implementation is very computationally intensive!
        about 30 minutes per 512x512 image
        """
        print('Calculating boundary weight...')
        n, H, W = target.shape
        weight = torch.zeros(n, H, W)
        ix, iy = np.meshgrid(np.arange(H), np.arange(W))
        ix, iy = np.c_[ix.ravel(), iy.ravel()].T
        for i, t in enumerate(tqdm(target)):
            boundary = find_boundaries(t, mode='inner')
            bound_x, bound_y = np.where(boundary is True)
            # broadcast boundary x pixel
            dx = (ix.reshape(1, -1) - bound_x.reshape(-1, 1)) ** 2
            dy = (iy.reshape(1, -1) - bound_y.reshape(-1, 1)) ** 2
            d = dx + dy
            # distance to 2 closest cells
            d2 = np.sqrt(np.partition(d, 2, axis=0)[:2, ])
            dsum = d2.sum(0).reshape(H, W)
            weight[i] = torch.Tensor(w0 * np.exp(-dsum**2 / (2 * sigma**2)))
        return weight


###############################################################################
# For testing
###############################################################################
def get_dataloader(mean, std, out_size, batch_size):
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

    data = CellDataset(
        image_mask_transform=image_mask_transform,
        # image_transform=image_transform,
        # mask_transform=mask_transform,
        data_type='train',
        weight_map_dir='weight_map.pt'
    )

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False
    )
    return loader


def visualize(image, mask):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(image)
    ax.set_title('Image')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(mask)
    ax.set_title('Label')

    plt.show()


if __name__ == '__main__':
    # print()
    mean = 0.495
    std = 0.173
    dataloader = get_dataloader(mean, std, 388, 2)

    for step, sample in enumerate(dataloader):
        X = sample['image']
        y = sample['mask']
        print(X.shape)
        # for i in range(X.shape[0]):
        #     image = X[i][0]
        #     mask = y[i][0]
        #     visualize(image, mask)

        break
