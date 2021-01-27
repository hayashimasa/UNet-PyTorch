# UNet-PyTorch

![alt text](https://github.com/hayashimasa/UNet-PyTorch/blob/main/visualization/UNet50.png?raw=true)

This is a PyTorch implementation of the U-Net architecture.

"U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015)

https://arxiv.org/pdf/1505.04597.pdf

# TODO
1. Include sample images for data augmentation
2. Use ResNet or other CNN architectures as encoder/decoder

# Introduction

U-Net is a fully convolutional neural network with an encoder-decoder structure designed for sementic image segmantation on biomedical images. [[1]](#1) It is a very effective meta-network architecture that has been adapted to incorporate other convolutional neural network architecture designs.

# Data

The data is from the 2012 ISBI challenge for segmentation of neuronal structures in electron microscopic stack. It is the same dataset used in the original paper.

The training data is a set of 30 sections from a serial section Transmission Electron Microscopy (ssTEM) dataset of the Drosophila first instar larva ventral nerve cord (VNC). The microcube measures 2 x 2 x 1.5 microns approx., with a resolution of 4x4x50 nm/pixel.[[2]](#2) Each image has 512x512 pixels.

The corresponding binary labels are provided in an in-out fashion, i.e. white for the pixels of segmented objects and black for the rest of pixels (which correspond mostly to membranes).

# Architecture

The network has a symmetric encoder-decoder structure. Images are first downsampled through a series of convolutional blocks consists of convolutional, non-linear activation, and max-pooling layers. The downsampled featured map is then symmetrically upsampled by a series of transposed convolutions in order to obatin a high resolution feature map that is close to the size of the original images. The most interesting feature of the architecture is the concatenation of high resolution feature maps from the contracting path and the corresponding upsampled feature maps from the expanding path. This design allows the network to leverage both high and low resolution information to learn the structure of the image. In order to increase efficiency and flexibility, a convolutional layer instead of a fully connected layer is used to output the final prediction. Each convolutional filter corresponding to an object classes.

![alt text](https://github.com/hayashimasa/UNet-PyTorch/blob/main/graphs/unet_architecture.png?raw=true)

The implementation of the network architecture is in `unet.py`.

# Data Augmentation

Data are scarced in the field of medical imaging (only 30 supervised image in this dataset); however, neural networks often rely on a large amount of supervised data to obtain good results; therefore, data augmentation is heavily utilized. The author suggests not only the typical affine transformation such as translation, rotation, and cropping, but also the use of elastic deformation. Deformation is a widely adopted technique for biomedical image segmentation tasks, since objects like cells and organs often have non-rigid structures.

Affine transformations and elastic deformation are simultaneously applied to both images and labels. Brightness adjustments and Gaussian noise are added to the input images.

During training, all augmentations are chosen stochastically; for each image and label, the augmentation is a composite of different combinations of transformations. For elastic deformation, the alpha parameter is chosen between 100-300, and the sigma parameter is chosen between 10-15.

The implementation of various data augmentation methods is in `augmentation.py`.

# Loss Function

Since this is a segmentic segmentation task, a pixel-loss is calculated through a softmax function combined with cross entropy over the final feature map.

<img src="https://render.githubusercontent.com/render/math?math=\large E = \sum_{x \in \Omega} w(x)log(p_{l(x)}(x))">

The loss function is implemented in `loss.py`.


Medical images often contain highly imbalanced object classes; therefore, the author suggests the use of a weighted loss. The weight function also incorporates the distance to the two closest cells and is defined by the following formula.

<img src="https://render.githubusercontent.com/render/math?math=\large w(x) = w_c(x) %2B w_0 \exp(-\frac{(d_1(x) %2B d_2(x))^2}{2\sigma^2})">

A vectorized implementation of the weighted function is in `celldata.py`.


# Training
```
python train.py --epoch 50 --batch-size 3 --save

Train image segmentation

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 3)
  --test-batch-size N   input batch size for testing (default: 3)
  --epochs N            number of epochs to train (default: 10)
  --lr LR               learning rate (default: 0.0005)
  --momentum M          SGD momentum (default: 0.5)
                        number of workers to load data
  --no-cuda             disables CUDA training
  --amp                 automatic mixed precision training
  --keep_batchnorm_fp32 KEEP_BATCHNORM_FP32
                        keep batch norm layers with 32-bit precision
  --log-interval N      how many batches to wait before logging training
                        status
  --save                save the current model
  --model MODEL         model to retrain
  --tensorboard         record training log to Tensorboard
```

The model was trained on 90% of the training data (27 images) and tested on 10% of the data (3 images) with the following hyperparameters:

Epoch: 50

Batch-size: 3

Optimizer: Adam

Learning rate: 0.0005

Objective function: Weighted Pixel-wise Cross-Entropy Loss

# Results

Intersection over Union (IOU): 0.758

Pixel Accuracy: 90.87%

![alt text](https://github.com/hayashimasa/UNet-PyTorch/blob/main/visualization/IOU.png?raw=true)

![alt text](https://github.com/hayashimasa/UNet-PyTorch/blob/main/visualization/pix_acc.png?raw=true)


IOU and training loss stagnate after roughly 30 epochs, and model reaches peak test performance at the 33rd epoch. Different batch sizes and learning rates were experimented to train the model for up to 50 more epochs, which is a total of 100 epochs. Training loss decreases but doesn't yield any improvement in segmentation performance; the model is likely overtraining.

![alt text](https://github.com/hayashimasa/UNet-PyTorch/blob/main/visualization/loss.png?raw=true)


# Reference

<a id="1">[1]</a>
Ronneberger, O., Fischer, P., & Brox, T. (2015).
U-Net: Convolutional Networks for Biomedical Image Segmentation.
MICCAI.

<a id="1">[2]</a>
Ignacio Arganda-Carreras, Srinivas C. Turaga, Daniel R. Berger, Dan Ciresan, Alessandro Giusti, Luca M. Gambardella, Jürgen Schmidhuber, Dmtry Laptev, Sarversh Dwivedi, Joachim M. Buhmann, Ting Liu, Mojtaba Seyedhosseini, Tolga Tasdizen, Lee Kamentsky, Radim Burget, Vaclav Uher, Xiao Tan, Chanming Sun, Tuan D. Pham, Eran Bas, Mustafa G. Uzunbas, Albert Cardona, Johannes Schindelin, and H. Sebastian Seung.
Crowdsourcing the creation of image segmentation algorithms for connectomics.
Frontiers in Neuroanatomy, vol. 9, no. 142, 2015.

# Project File Organization
```
UNet-PyTorch
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
    ├── IOU.png
    ├── UNet50.png
    ├── loss.png
    └── pix_acc.png
```
