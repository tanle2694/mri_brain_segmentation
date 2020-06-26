import torch
import numpy as np


class IoU(object):

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, )*2)

    def get_iou(self):
        IoU_value = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) +
                                                 np.sum(self.confusion_matrix, axis=0)- np.diag(self.confusion_matrix))
        # mIoU = np.nanmean(IoU)
        iou_class = {}
        for i in range(IoU_value.shape[0]):
            iou_class["iou_class_{}".format(i)] = IoU_value[i]
        return iou_class

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


def dice_coeff(predict, target, threshold=0.5):
    N = target.size()
    predict[predict >= threshold] = 1
    predict[predict < threshold] = 0

    predict_flat = predict.reshape()


