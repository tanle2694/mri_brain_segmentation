import torch
import random
import numpy as np

from PIL import Image


class Normalize(object):
    def __init__(self, mean=[0., 0., 0.], std=[1., 1., 1.]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        img_float = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img_norm = img_float / 255.0
        img_norm -= self.mean
        img_norm /= self.std
        return {'image': img_norm, 'mask':  mask}


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        img_float = np.array(image).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img_tensor = torch.from_numpy(img_float).float()
        mask_tensor = torch.from_numpy(mask).float()

        return {'image': img_tensor, 'mask': mask_tensor}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': image, 'mask': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        random_rot_degree = random.uniform(-1 * self.degree, self.degree)
        image_rot = image.rotate(random_rot_degree, Image.BILINEAR)
        mask = mask.rotate(random_rot_degree, Image.BILINEAR)
        return {'image': image_rot, 'mask': mask}

