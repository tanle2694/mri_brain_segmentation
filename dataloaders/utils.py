import torch
import random
import numpy as np
from PIL import Image


def add_margin(pil_img, width_padding, height_padding, color):
    width, height = pil_img.size
    new_width = width + width_padding
    new_height = height + height_padding
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (0, 0))
    return result


class Padding(object):
    def __init__(self, input_size, padding_value_origin, padding_value_mask):
        self.input_size = input_size
        self.padding_value_origin = padding_value_origin
        self.padding_value_mask = padding_value_mask

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        w, h = image.size
        image = add_margin(image, self.input_size - w, self.input_size - h, self.padding_value_origin)
        mask = add_margin(mask, self.input_size - w, self.input_size - h, self.padding_value_mask)
        return {"image": image, "mask": mask}


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


class RandomScaleCrop(object):
    def __init__(self, range_scale, crop_size):
        self.range_scale = range_scale
        self.crop_size = crop_size

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        scale_ratio = random.uniform(*self.range_scale)
        w, h = image.size
        w_resize, h_resize = int(w * scale_ratio), int(h * scale_ratio)
        image = image.resize((w_resize, h_resize), Image.BILINEAR)
        mask = mask.resize((w_resize, h_resize), Image.NEAREST)
        if (w_resize <= self.crop_size) or (h_resize <= self.crop_size):
            return {"image": image, "mask": mask}

        x_crop = (w_resize - self.crop_size) // 2
        y_crop = (h_resize - self.crop_size) // 2
        image = image.crop((x_crop, y_crop, x_crop + self.crop_size, y_crop + self.crop_size))
        mask = mask.crop((x_crop, y_crop, x_crop + self.crop_size, y_crop + self.crop_size))
        return {"image": image, "mask": mask}
