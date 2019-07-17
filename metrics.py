import torch.nn as nn
import torch
import numpy as np
from dataprocess import to_categorical


class IoUMetric(nn.Module):

    __name__ = 'iou'

    def __init__(self, eps=1e-7, threshold=0.5, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps
        self.threshold = threshold

    def forward(self, y_pr, y_gt):
        return iou(y_pr, y_gt, self.eps, self.threshold, self.activation)

def iou(pr, gt, eps=1e-7, threshold=None, activation='sigmoid'):

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)
    iou_all = 0
    smooth = 1
    pr = torch.argmax(pr, dim=1)
    pr = pr.cpu().numpy()
    gt = gt.cpu().numpy()

    pr = to_categorical(pr, num_classes=3)
    gt = to_categorical(gt, num_classes=3)
    nb_classes = 3
    for i in range(0, nb_classes):
        res_true = gt[:, :, :, i:i + 1]
        res_pred = pr[:, :, :, i:i + 1]

        res_pred = res_pred.astype(np.float64)
        res_true = res_true.astype(np.float64)

        intersection = np.sum(np.abs(res_true * res_pred), axis=(1, 2, 3))
        union = np.sum(res_true, axis=(1, 2, 3)) + np.sum(res_pred, axis=(1, 2, 3)) - intersection
        iou_all += (np.mean((intersection + smooth) / (union + smooth), axis=0))

    return iou_all / nb_classes

