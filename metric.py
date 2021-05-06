"""Evaluation Metric

author: Masahiro Hayashi

This script defines metrics to evaluate model performance.
"""
import torch
import numpy as np


def pix_acc(outputs, targets, batch_size):
    """Pixel accuracy

    Args:
        outputs (torch.nn.Tensor): prediction outputs
        targets (torch.nn.Tensor): prediction targets
        batch_size (int): size of minibatch
    """
    acc = 0.
    for idx in range(batch_size):
        output = outputs[idx]
        target = targets[idx]
        correct = torch.sum(torch.eq(output, target).long())
        acc += correct / np.prod(np.array(output.shape)) / batch_size
    return acc.item()


def iou(outputs, targets, batch_size, n_classes):
    """Intersection over union

    Args:
        outputs (torch.nn.Tensor): prediction outputs
        targets (torch.nn.Tensor): prediction targets
        batch_size (int): size of minibatch
        n_classes (int): number of segmentation classes
    """
    eps = 1e-6
    class_iou = np.zeros(n_classes)
    for idx in range(batch_size):
        outputs_cpu = outputs[idx].cpu()
        targets_cpu = targets[idx].cpu()

        for c in range(n_classes):
            i_outputs = np.where(outputs_cpu == c)  # indices of 'c' in output
            i_targets = np.where(targets_cpu == c)  # indices of 'c' in target
            intersection = np.intersect1d(i_outputs, i_targets).size
            union = np.union1d(i_outputs, i_targets).size
            class_iou[c] += (intersection + eps) / (union + eps)

    class_iou /= batch_size

    return class_iou


###############################################################################
# testing
###############################################################################
if __name__ == '__main__':
    x = torch.randint(high=2, size=(3, 1, 5, 5))
    y = torch.randint(high=2, size=(3, 1, 5, 5))
    print(x[0])
    print(y[0])
    print(torch.sum(torch.eq(x[0], y[0]).long()))
    print(sum(x[0].shape[1:]))

    print(pix_acc(x, y, 3))
