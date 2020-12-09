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
        '--model', type=str, default=None,
        help='model to use for inference'
    )
    args = parser.parse_args()
    return args

def get_test_loader(mean, std, out_size, batch_size):
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
        data_type='test'
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False#,
        # num_workers=args.num_workers
    )
    return test_loader

def validate(model, device, data_loader, criterion, n_classes):
    model.eval()
    test_loss = 0
    # n = len(data_loader.dataset)
    # class_iou = [0.] * n_classes
    # pixel_acc = 0.
    pred = None
    with torch.no_grad():
        for sample in data_loader:
            X = sample['image'].to(device)
            y = sample['mask'].to(device)
            y = y.squeeze(1) # remove channel dimension
            y_pred = model(X)
            # test_loss += criterion(y_pred, y.long()).item() # sum up batch loss

            pred = torch.argmax(y_pred, dim=1)

    return pred

if __name__ == '__main__':
    args = parse_args()
    path = os.getcwd() + '/data/test-volume.tif'
    image = io.imread(path)

    print(image.shape)

    im0 = image[1]

    # plt.imshow(im)
    # plt.show()
    mean = 0.495
    std = 0.173
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Pad(30, padding_mode='reflect')
    ])
    im = image_transform(im0)
    print(im.shape)
    im = im.view(1, *im.shape)
    model_name = args.model if args.model else 'UNet25.pt'
    checkpoint_path = os.getcwd() + f'/models/{model_name}'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = UNet(2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    y_pred = model(im)
    pred = torch.argmax(y_pred, dim=1)[0]
    print(pred.shape)
    dim = im0.shape
    out_size = pred.shape[0]
    cut = (dim[0] - out_size) // 2
    im0 = im0[cut:cut+out_size, cut:cut+out_size]
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(im0)
    ax.set_title('Image')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(pred)
    ax.set_title('Prediction')
    plt.savefig(f'visualization/{args.model[:-3]}.png')
    plt.show()
