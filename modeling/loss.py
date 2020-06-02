import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

def cross_entropy(logit, target):
    n, _, _, _ = logit.size()
    criterion = nn.CrossEntropyLoss(ignore_index=-1,
                                    size_average=True)
    criterion = criterion.cuda()
    loss = criterion(logit, target.long())
    loss /= n
    return loss


def dice_loss(y_pred, y_true):
    # y_pred = y_pred.argmax(1).type(torch.FloatTensor).cuda("cuda:1")
    # y_pred = Variable(torch.argmax(y_pred, 1).float(), requires_grad=True).cuda("cuda:1")

    y_pred = F.sigmoid(y_pred)[:, 0].contiguous()
    assert y_pred.size() == y_true.size()
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    dsc = (2. * intersection + 1) / (
        y_pred.sum() + y_true.sum() + 1
    )
    return 1. - dsc
