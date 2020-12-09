# UNet-PyTorch

This is a PyTorch implementation of the U-Net architecture.

"U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015)

https://arxiv.org/pdf/1505.04597.pdf

# Introduction

U-Net is a fully convolutional neural network with an encoder-decoder structure. It is designed for semantic segmantation of biomedical images.

# Data

The data is taken from the 2012 ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. It is the same data used in the original paper.

The training data is a set of 30 sections from a serial section Transmission Electron Microscopy (ssTEM) data set of the Drosophila first instar larva ventral nerve cord (VNC). The microcube measures 2 x 2 x 1.5 microns approx., with a resolution of 4x4x50 nm/pixel. Each cell image has 512x512 pixels.

The corresponding binary labels are provided in an in-out fashion, i.e. white for the pixels of segmented objects and black for the rest of pixels (which correspond mostly to membranes).

The data is contained in `data/`.

# Architecture

U-Net has an encoder-decoder structure. An image is first downsampled through a series of convolutional blocks that each consists of convolutional, non-linear activation, and maxpooling layers. The downsampled feature map is then symmetrically upsampled through a series of transposed convolution blocks in order to retrieve a feature map with a size that is close to the original image resolution. The final featured map is mapped to the desired segmentation classes through a convolutional layer where the filter size equals the number of segmentation classes. The most interesting part about the U-Net architecture is the concatenation of feature map from the downsampling path with the corresponding feature map in the upsampling path. This allows the network to learn from both the high resolution features from the contracting path and the upsampled features at the same time.

![alt text](https://github.com/hayashimasa/UNet-PyTorch/blob/main/image/unet_architecture.png?raw=true)

The model is implemented in `unet.py`

# Data Augmentation

Data are scarced in the field of medical imaging; therefore, data augmentation is crucial for efficient training. Besides the usual affine transformation such as fliping and cropping, the authors recommend the use of elastic deformation, which is a method often utilized in biomedical segmentation. A 30-pixel reflective padding is added on each side in order to output the desired region of segmentation, which in the authors' case is a 388x388 segmentation map.

The various data augmentation is implemented in `augmentation.py`.

# Loss Function

Biomedical images often consists highly imbalanced object classes, so the authors suggest a loss function that incoporates both the class imbalance and the distance to nearest cells. Since this is a semantic segmentation problem, the network is trained through a pixel-wise loss that takes a softmax over the final feature map combined with cross entropy.

<img src="https://render.githubusercontent.com/render/math?math=\large \sum_{x\in \Omega} w(x) log(p_{l(x)}(x))">

The weight is calculated by the following formula:

<img src="https://render.githubusercontent.com/render/math?math=\large w(x) = w_c(x) %2B w_0 \dot \exp(-\frac{(d_1(x) %2B d_2(x))^2}{2\sigma^2})">

A vectorized implementation of the weight map is in `celldata.py`.

```
.
├── LICENSE
├── README.md
├── data
│   ├── test-volume.tif
│   ├── train-labels.tif
│   └── train-volume.tif
├── models
│   └── UNet100.pt
├── visualization
│   └── unet100.png
├── augmentation.py
├── celldata.py
├── inference.py
├── loss.py
├── metric.py
├── train.py
├── unet.py
└── weight_map.pt
```


