# -*- coding: utf-8 -*-
"""U-Net

author: Masahiro Hayashi

This script defines the model architecture of U-Net.
"""

import torch
import torch.nn as nn


class UNet(torch.nn.Module):
    """Implementation of the U-Net architecture
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    by Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015)
    https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self, n_classes, batch_norm=True):
        """
        """
        self.name = 'UNet'
        self.n_classes = n_classes
        self.filter_sizes = [64, 128, 256, 512, 1024]
        self.n_block = len(self.filter_sizes)
        self.batch_norm = batch_norm
        # self.num_layers = [2, 2, 2, 2, 2]

        super(UNet, self).__init__()
        self.contract_blocks = self.contract()
        self.expand_blocks = self.expand()
        self.segment = nn.Conv2d(
            self.filter_sizes[0],
            self.n_classes,
            kernel_size=1
        )

    def forward(self, x):
        """Performs a forward pass through the network
        """
        xs = []
        for block in self.contract_blocks:
            new_x = block(x)
            xs.append(new_x)
            x = new_x

        for i, block in enumerate(self.expand_blocks):
            x = block['up'](x)
            k = self.n_block - i - 2
            x = self.concat(xs[k], x)
            x = block['conv'](x)

        y_pred = self.segment(x)

        return y_pred

    def concat(self, x, y):
        """Crop and concatenate two feature maps
        """
        dw = (x.size()[2] - y.size()[2]) // 2
        dh = (x.size()[3] - y.size()[2]) // 2
        x = x[:, :, dw:x.size()[2] - dw, dh:x.size()[3] - dh]
        return torch.cat((x, y), 1)

    def contract(self):
        """Define contraction block in U-Net
        """
        blocks = []
        old = 1
        for i, size in enumerate(self.filter_sizes):
            mpool = nn.MaxPool2d(kernel_size=2)
            conv1 = nn.Conv2d(old, size, kernel_size=3)
            conv2 = nn.Conv2d(size, size, kernel_size=3)
            relu = nn.ReLU(True)
            convs = [mpool, conv1, relu, conv2, relu]
            if self.batch_norm:
                b_norm = nn.BatchNorm2d(size)
                convs = [mpool, conv1, b_norm, relu, conv2, b_norm, relu]
            if i == 0:
                convs = convs[1:]
            block = nn.Sequential(*convs)
            blocks.append(block)
            old = size
            self.add_module(f'contract{i+1}', block)
        return blocks

    def expand(self):
        """Define expansion block in U-Net
        """
        blocks = []
        expand_filters = self.filter_sizes[self.n_block - 2::-1]
        old = self.filter_sizes[-1]
        for i, size in enumerate(expand_filters):
            up = nn.ConvTranspose2d(old, size, kernel_size=2, stride=2)
            self.add_module(f'up{i+1}', up)
            conv1 = nn.Conv2d(old, size, kernel_size=3)
            conv2 = nn.Conv2d(size, size, kernel_size=3)
            relu = nn.ReLU(True)
            convs = [conv1, relu, conv2, relu]
            if self.batch_norm:
                b_norm = nn.BatchNorm2d(size)
                convs = [conv1, b_norm, relu, conv2, b_norm, relu]
            convs = nn.Sequential(*convs)
            self.add_module(f'deconv{i+1}', convs)
            blocks.append({'up': up, 'conv': convs})

            old = size

        return blocks


###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 1, 572, 572)
    model = UNet(2)

    print(list(model.children()))
    import time
    t = time.time()
    x = model(im)
    print(time.time() - t)
    print(x.shape)
    del model
    del x
