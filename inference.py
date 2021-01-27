"""Model Inference

author: Masahiro Hayashi

This script allows users to make inference and visualize results.
"""
import os
import argparse
import torch
from torchvision import transforms
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from unet import UNet

def parse_args():
    parser = argparse.ArgumentParser(
        description='Make segmentation predicitons'
    )
    parser.add_argument(
        '--model', type=str, default='UNet50.pt',
        help='model to use for inference'
    )
    parser.add_argument(
        '--visualize', action='store_true', default=False,
        help='visualize the inference result'
    )
    args = parser.parse_args()
    return args

def predict(image, model):
    """Make prediction on image"""
    mean = 0.495
    std = 0.173
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Pad(30, padding_mode='reflect')
    ])
    im = image_transform(image)
    im = im.view(1, *im.shape)
    model.eval()
    y_pred = model(im)
    pred = torch.argmax(y_pred, dim=1)[0]
    return pred

def visualize(image, pred, label=None):
    """make visualization"""
    n_plot = 2 if label is None else 3
    fig = plt.figure()
    ax = fig.add_subplot(1, n_plot, 1)
    imgplot = plt.imshow(image)
    ax.set_title('Image')
    ax = fig.add_subplot(1, n_plot, 2)
    imgplot = plt.imshow(pred)
    ax.set_title('Prediction')
    if n_plot > 2:
        ax = fig.add_subplot(1, n_plot, 3)
        imgplot = plt.imshow(label)
        ax.set_title('Ground Truth')
    fig.tight_layout()
    plt.savefig(f'visualization/{args.model[:-3]}_validation.png')
    plt.show()

if __name__ == '__main__':
    args = parse_args()

    # load images and labels
    path = os.getcwd() + '/data/train-volume.tif'
    images = io.imread(path)
    label_path = os.getcwd() + '/data/train-labels.tif'
    labels = io.imread(label_path)
    image = images[-1]
    label = labels[-1]

    # load model
    checkpoint_path = os.getcwd() + f'/models/{args.model}'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = UNet(2)
    model.load_state_dict(checkpoint['model_state_dict'])
    # make inference
    pred = predict(image, model)

    if args.visualize:
        # crop images for visualization
        dim = image.shape
        out_size = pred.shape[0]
        cut = (dim[0] - out_size) // 2
        image = image[cut:cut+out_size, cut:cut+out_size]
        label = label[cut:cut+out_size, cut:cut+out_size]
        # visualize result
        visualize(image, pred, label)


