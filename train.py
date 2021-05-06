"""Model Training

author: Masahiro Hayashi

This script defines the training process of the segmentation model.
"""
import os
import argparse

import torch
from torch import nn, optim, DoubleTensor
from torch import cuda
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, datasets
import numpy as np

from unet import UNet
from celldata import CellDataset
from metric import iou, pix_acc
from loss import Weighted_Cross_Entropy_Loss
from augmentation import (
    DoubleCompose, DoubleToTensor,
    DoubleHorizontalFlip, DoubleVerticalFlip, DoubleElasticTransform,
    GaussianNoise
)
# from apex import amp, optimizers


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument(
        '--batch-size', type=int, default=3, metavar='N',
        help='input batch size for training (default: 3)'
    )
    parser.add_argument(
        '--test-batch-size', type=int, default=3, metavar='N',
        help='input batch size for testing (default: 3)'
    )
    parser.add_argument(
        '--epochs', type=int, default=10, metavar='N',
        help='number of epochs to train (default: 10)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0001, metavar='LR',
        help='learning rate (default: 0.0001)'
    )
    parser.add_argument(
        '--momentum', type=float, default=0.5, metavar='M',
        help='SGD momentum (default: 0.5)'
    )
    parser.add_argument(
        '--n-classes', type=int, default=2,
        help='number of segmentation classes'
    )
    parser.add_argument(
        '--num_workers', type=int, default=6,
        help='number of workers to load data'
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='disables CUDA training'
    )
    parser.add_argument(
        '--amp', action='store_true', default=False,
        help='automatic mixed precision training'
    )
    parser.add_argument(
        '--opt-level', type=str
    )
    parser.add_argument(
        '--keep_batchnorm_fp32', type=str, default=None,
        help='keep batch norm layers with 32-bit precision'
    )
    parser.add_argument(
        '--loss-scale', type=str, default=None
    )
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)'
    )
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging training status'
    )
    parser.add_argument(
        '--save', action='store_true', default=False,
        help='save the current model'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='model to retrain'
    )
    parser.add_argument(
        '--tensorboard', action='store_true', default=False,
        help='record training log to Tensorboard'
    )
    args = parser.parse_args()
    return args


def get_train_loader(mean, std, out_size, batch_size, pct=.9):
    """Initialize Dataloader for training set

        mean (float): mean of pixel values
        std (float): standard deviation of pixel values
        out_size (int): dimension of segmentation map (out_size x out_size)
        batch_size (int): number of samples to load for each iteration
        pct (float): percentage of data to use for training (0 < pct <= 1)
    """
    image_mask_transform = DoubleCompose([
        DoubleToTensor(),
        DoubleElasticTransform(alpha=250, sigma=10),
        DoubleHorizontalFlip(),
        DoubleVerticalFlip()
    ])
    image_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.4),
        transforms.Normalize(mean, std),
        GaussianNoise(),
        transforms.Pad(30, padding_mode='reflect')
    ])
    mask_transform = transforms.CenterCrop(out_size)

    train_data = CellDataset(
        image_mask_transform=image_mask_transform,
        image_transform=image_transform,
        mask_transform=mask_transform,
        pct=pct
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader


def get_test_loader(mean, std, out_size, batch_size):
    """Initialize Dataloader for validation set

        mean (float): mean of pixel values
        std (float): standard deviation of pixel values
        out_size (int): dimension of segmentation map (out_size x out_size)
        batch_size (int): number of samples to load for each iteration
    """
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Pad(30, padding_mode='reflect')
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(388)
    ])

    test_data = CellDataset(
        image_transform=image_transform,
        mask_transform=mask_transform,
        data_type='validate'
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )
    return test_loader


def train(model, device, data_loader, optimizer, criterion, epoch):
    """train model for one epoch

    Args:
        model (torch.nn.Module): model to train
        device (str): device to train model ('cpu' or 'cuda')
        data_loader (object): iterator to load data
        optimizer (torch.nn.optim): stochastic optimzation strategy
        criterion (torch.nn.Module): loss function
        epoch (int): current epoch
    """
    model.train()
    loss = 0.
    for step, sample in enumerate(data_loader):
        # forward pass
        X = sample['image'].to(device)
        y = sample['mask'].to(device)
        w = sample['weight'].to(device)
        y = y.squeeze(1).long()  # remove channel dimension
        y_pred = model(X)

        # back propogation
        loss = criterion(y_pred, y, w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        log_interval = 1
        if step % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (step+1) * len(X), len(data_loader.dataset),
                    100. * (step+1) / len(data_loader), loss.item()
                )
            )
        # break

    return loss.item()


def validate(model, device, data_loader, criterion, n_classes):
    """Evaluate model performance with validation data

    Args:
        model (torch.nn.Module): model to evaluate
        device (str): device to evaluate model ('cpu' or 'cuda')
        data_loader (object): iterator to load data
        criterion (torch.nn.Module): loss function
        n_classes (int): number of segmentation classes
    """
    model.eval()
    test_loss = 0
    n = len(data_loader.dataset)
    class_iou = [0.] * n_classes
    pixel_acc = 0.
    with torch.no_grad():
        for sample in data_loader:
            X = sample['image'].to(device)
            y = sample['mask'].to(device)
            w = sample['weight'].to(device)
            y = y.squeeze(1).long()  # remove channel dimension
            y_pred = model(X)
            test_loss += criterion(y_pred, y, w).item()  # sum up batch loss
            pred = torch.argmax(y_pred, dim=1)
            batch_size = X.shape[0]
            pred = pred.view(batch_size, -1)
            y = y.view(batch_size, -1)
            batch_iou = iou(pred, y, batch_size, n_classes)
            class_iou += batch_iou * (batch_size / n)
            pixel_acc += pix_acc(pred, y, batch_size) * (batch_size / n)

    data_size = len(data_loader.dataset)
    test_loss /= data_size
    avg_iou = np.mean(class_iou)

    print(
        '\nValidation set: Average loss: {:.4f}, '.format(test_loss)
        + 'Average IOU score: {:.2f}, '.format(avg_iou)
        + 'Average pixel accuracy: {:.2f}\n'.format(pixel_acc)
    )
    return test_loss, avg_iou, pixel_acc


def initialize_model(args):
    """Initialize model checkpoint dictionary for storing training progress

    Args:
        args (object):
            epoch (int): total number of epochs to train model
            n_classes (int): number of segmentation classes
    """
    model_dict = {
        'total_epoch': args.epochs,
        'n_classes': args.n_classes,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'train_loss': list(),
        'test_loss': list(),
        'metrics': {
            'IOU': list(),
            'pix_acc': list(),
            'best': {
                'IOU': 0.,
                'pixel_acc': 0.,
                'epoch': 0
            }
        }
    }
    return model_dict


def get_model(args, device):
    """Intialize or load model checkpoint and intialize model and optimizer

    Args:
        args (object):
            model (str): filename of model to load
                (initialize new model if none is given)
        device (str): device to train and evaluate model ('cpu' or 'cuda')
    """
    if args.model:
        # Load model checkpoint
        model_path = os.path.join(os.getcwd(), f'models/{args.model}')
        model_dict = torch.load(model_path)
    else:
        model_dict = initialize_model(args)
    n_classes = model_dict['n_classes']
    model = UNet(n_classes).cuda() if device == 'cuda' else UNet(n_classes)
    optimizer = optim.Adam(model.parameters(), args.lr)
    if args.model:
        model.load_state_dict(model_dict['model_state_dict'])
        optimizer.load_state_dict(model_dict['optimizer_state_dict'])
    return model, optimizer, model_dict


if __name__ == '__main__':
    args = parse_args()
    if args.tensorboard:
        writer = SummaryWriter()
    # initialize model
    device = (
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    model, optimizer, model_dict = get_model(args, device)
    # initialize dataloader
    mean = 0.495
    std = 0.173
    out_size = 388  # output dimension of segmentation map
    train_loader = get_train_loader(mean, std, out_size, args.batch_size)
    test_loader = get_test_loader(mean, std, out_size, args.test_batch_size)
    # define loss function
    criterion = Weighted_Cross_Entropy_Loss()
    # train and evaluate model
    start_epoch = 1 if not args.model else model_dict['total_epoch'] + 1
    n_epoch = start_epoch + args.epochs - 1
    model_path = os.getcwd() + '/models'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    model_name = f'models/{model.name}{n_epoch}.pt'
    for epoch in range(start_epoch, n_epoch + 1):
        train_loss = train(
            model, device, train_loader, optimizer, criterion, epoch
        )
        test_loss, test_iou, test_pix_acc = validate(
            model, device, test_loader, criterion, args.n_classes
        )
        # update tensorboard
        if args.tensorboard:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('IOU/test', test_iou, epoch)
            writer.add_scalar('Pixel_Accuracy/test', test_pix_acc, epoch)
        # record training progress
        model_dict['train_loss'].append(train_loss)
        model_dict['test_loss'].append(test_loss)
        model_dict['metrics']['IOU'].append(test_iou)
        model_dict['metrics']['pix_acc'].append(test_pix_acc)
        if epoch == 1 or test_iou > model_dict['metrics']['best']['IOU']:
            model_dict['model_state_dict'] = model.state_dict()
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
            model_dict['metrics']['best']['IOU'] = test_iou
            model_dict['metrics']['best']['pix_acc'] = test_pix_acc
            model_dict['metrics']['best']['epoch'] = epoch
        if args.save:
            torch.save(model_dict, model_name)
    if args.tensorboard:
        writer.close()
    # print model statistics
    # print('training loss:', train_losses)
    # print('validation loss:', test_losses)
    # print('Intersection over Union:', test_ious)
    print('Best IOU:', model_dict['metrics']['best']['IOU'])
    print('Pixel accuracy:', model_dict['metrics']['best']['pix_acc'])
