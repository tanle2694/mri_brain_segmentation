import torch
import numpy as np

class MIoU(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, )*2)

    def get_miou(self):
        IoU = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) +
                                                 np.sum(self.confusion_matrix, axis=0)- np.diag(self.confusion_matrix))
        mIoU = np.nanmean(IoU)
        return mIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

