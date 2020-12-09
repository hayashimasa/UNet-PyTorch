# UNet-PyTorch
<<<<<<< HEAD

This is a PyTorch implementation of the U-Net architecture.

"U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015)

https://arxiv.org/pdf/1505.04597.pdf

# TODO
1. Add results
2. Include sample image for data augmentation
3. Fix custom loss implementation

# Introduction

U-Net is a fully convolutional neural network with an encoder-decoder structure designed for sementic image segmantation on biomedical images. It is a very effective meta-network architecture that has been adapted to incorporate other convolutional neural network architecture designs.

# Data

The data is from the 2012 ISBI challenge for segmentation of neuronal structures in electron microscopic stack. It is the same dataset used in the orginal paper.

The training data is a set of 30 sections from a serial section Transmission Electron Microscopy (ssTEM) data set of the Drosophila first instar larva ventral nerve cord (VNC). The microcube measures 2 x 2 x 1.5 microns approx., with a resolution of 4x4x50 nm/pixel. Each image has 512x512 pixels.

The corresponding binary labels are provided in an in-out fashion, i.e. white for the pixels of segmented objects and black for the rest of pixels (which correspond mostly to membranes).

# Architecture

The network has a symmetric encoder-decoder structure. Images are first downsampled through a series of convolutional blaocks consists of convolutional, non-linear activation, and max-pooling layers. The downsampled featured map is then symmetrically upsampled by a series of transposed convolutions in order to obatin a high resolution feature map that is close to the size of the original images. The most interesting feature of the architecture is the concatenation of high resolution feature maps from the contracting path and the corresponding upsampled feature maps from the expanding path. This design allows the network to leverage both high and low resolution information to learn the structure of the image. In order to increase efficiency and flexibility, a convolutional layer instead of a fully connected layer is used to output the final prediction. Each convolutional filter corresponding to an object classes.

![alt text](https://github.com/hayashimasa/UNet-PyTorch/blob/main/graphs/Unet100.png?raw=true)

The implementation of the network architecture is in `unet.py`.

# Data Augmentation

Data are scarced in the field of medical imaging (only 30 supervised image in this dataset); however, neural networks rely on a large amount of supervised data; therefore, data augmentation is heavily utilized. The author suggests not only the typical affine transformation such as translation, rotation, and cropping, but also the use of elastic deformation. Deformation is a widely adopted technique for biomedical image segmentation tasks.

The implementation of various data augmentation methods is in `augmentation.py`.

# Loss Function

Since this is a segmentic segmentation task, a pixel-loss is calculated through a softmax function combined with cross entropy over the final feature map.

<img src="https://render.githubusercontent.com/render/math?math=\large E = \sum_{x \in \Omega} w(x)log(p_{l(x)}(x))">

The loss function is implemented in `loss.py`.


Medical images often contain highly imbalanced object classes; therefore, the author suggests the use of a weighted loss. The weight function also incorporates the distance to the closest cell boundary and is defined by the following formula.

<img src="https://render.githubusercontent.com/render/math?math=\large w(x) = w_c(x) %2B w_0 \exp(-\frac{(d_1(x) %2B d_2(x))^2}{2\sigma^2})">

A Vectorized implementation of the weighted function is in `celldata.py`.


# Project File Organization
```
.
├── LICENSE
├── README.md
├── data
│   ├── test-volume.tif
│   ├── train-labels.tif
│   └── train-volume.tif
├── celldata.py
├── augmentation.py
├── unet.py
├── train.py
├── loss.py
├── metric.py
├── inference.py
└── visualization
    └── unet100_0.png
```
<!-- # Results -->
=======
PyTorch implementation of the U-Net architecture
>>>>>>> fc8af7b37b4300ebee4d700400927262d70283a6
