import torch.nn.functional as F
import torch.nn as nn

def cross_entropy(logit, target):
    n, _, _, _, _ = logit.size()
    criterion = nn.CrossEntropyLoss(ignore_index=-1,
                                    size_average=True)
    criterion = criterion.cuda()
    loss = criterion(logit, target.long())
    loss /= n
    return loss
