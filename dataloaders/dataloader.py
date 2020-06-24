import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
from pathlib import Path
from dataloaders.utils import Normalize, ToTensor, RandomHorizontalFlip, RandomRotate, Padding, RandomScaleCrop
import cv2
import numpy as np


class MRIBrainSegmentation(data.Dataset):
    def __init__(self, root_folder, image_label, is_train=True, ignore_label=0, input_size=300):
        super(MRIBrainSegmentation, self).__init__()
        root_folder = Path(root_folder)
        image_label = Path(image_label).read_text().split("\n")
        self.images = []
        self.labels = []
        for i_l in image_label:
            if len(i_l) == 0:
                continue
            image, label = i_l.strip().split("\t")
            self.images.append(root_folder / image)
            self.labels.append(root_folder / label)
        self.transform = transforms
        self.is_train = is_train
        self.ignore_label = ignore_label
        self.input_size = input_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_link = self.images[index]

        label_link = self.labels[index]
        _img = Image.open(img_link).convert('RGB')
        # img_width, img_height = _img.size
        # _img = add_margin(_img, self.input_size - img_width, self.input_size - img_height, (0, 255, 0))

        _target = Image.open(label_link).convert('RGB')
        # target_width, target_height = _target.size
        # _target = add_margin(_target, self.input_size - target_width, self.input_size - target_height, (255, 255, 255))

        _target_numpy = np.array(_target)[:, :, 0]
        _target_numpy[_target_numpy == 255] = 1
        # _target_numpy[target_height:, :] = 255
        # _target_numpy[:, target_width:] = 255
        _target = Image.fromarray(_target_numpy)

        sample = {"image": _img, "mask": _target}
        if self.is_train:
            compose_transform = transforms.Compose([
                RandomScaleCrop(range_scale=(0.5, 2.0), crop_size=self.input_size),
                RandomHorizontalFlip(),
                RandomRotate(30),
                # Padding(input_size=self.input_size, padding_value_origin=(124, 117, 104), padding_value_mask=255),
                Padding(input_size=self.input_size, padding_value_origin=(0, 0, 0), padding_value_mask=self.ignore_label),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor()
            ])
            return compose_transform(sample)
        compose_transform = transforms.Compose([
            Padding(input_size=self.input_size, padding_value_origin=(124, 117, 104), padding_value_mask=self.ignore_label),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])
        return compose_transform(sample)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2

    dataset = MRIBrainSegmentation(root_folder="/home/tanlm/Downloads/data/kaggle_3m",
                                   image_label="/home/tanlm/Downloads/data/vali.txt",
                                   is_train=True, ignore_label=0)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for index, sample in enumerate(dataloader):

        for i in range(sample['image'].size()[0]):
            img = sample['image'].numpy()[i].transpose([1, 2, 0])
            mask = sample['mask'].numpy()[i]
            img *= (0.229, 0.224, 0.225)
            img += (0.485, 0.456, 0.406)
            img *= 255.0
            img = img.astype(np.uint8)
            mask = mask.astype(np.uint8)
            # mask_3ch = np.zeros((mask.shape[0], mask.shape[1], 3))
            # mask_3ch[np.where(mask == 0)] = [0, 255, 0]
            # mask_3ch[np.where(mask == 1)] = [0, 0, 255]
            # mask_3ch[np.where(mask == 255)] = [255, 0 , 0]
            contours, hierarchy = cv2.findContours(mask,
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            img_draw = cv2.drawContours(cv2.UMat(img), contours, -1, (0, 0, 255), 1)
            cv2.imshow("img", img)
            cv2.imshow("draw", img_draw)
            cv2.imshow("mask", (mask * 255).astype(np.uint8))
            cv2.waitKey(0)
