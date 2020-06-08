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

    # y_pred = F.sigmoid(y_pred)[:, 0].contiguous()
    y_pred = y_pred[:,0]
    print(y_pred.size(), y_true.size())
    assert y_pred.size() == y_true.size()
    y_pred = y_pred.view(y_pred.size()[0], -1)
    y_true = y_true.view(y_true.size()[0], -1)
    intersection = (y_pred * y_true).sum(dim=1)
    dsc = (2. * intersection + 1) / (
        y_pred.sum(dim=1) + y_true.sum(dim=1) + 1
    )
    dsc = torch.mean(dsc)
    return 1. - dsc
