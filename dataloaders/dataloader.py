import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
from pathlib import Path
from dataloaders.utils import Normalize, ToTensor, RandomHorizontalFlip, RandomRotate
import cv2

class MRIBrainSegmentation(data.Dataset):
    def __init__(self, root_folder, image_label, is_train=True):
        super(MRIBrainSegmentation, self).__init__()
        root_folder = Path(root_folder)
        image_label = Path(image_label).read_text().split("\n")
        self.images = []
        self.labels = []
        for i_l in image_label:
            image, label = i_l.strip().split("\t")
            self.images.append(root_folder / image)
            self.labels.append(root_folder / label)
        self.transform = transforms
        self.is_train = is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_link = self.images[index]
        label_link = self.labels[index]
        _img = Image.open(img_link).convert('RGB')
        _target = Image.open(label_link).convert('RGB')
        _target_numpy = np.array(_target)[:, :, 0]
        _target_numpy[_target_numpy == 255] = 1
        _target = Image.fromarray(_target_numpy)

        sample = {"image": _img, "mask": _target}
        if self.is_train:
            compose_transform = transforms.Compose([
                RandomHorizontalFlip(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor()
            ])
            return compose_transform(sample)
        compose_transform = transforms.Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])
        return compose_transform(sample)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2

    dataset = MRIBrainSegmentation(root_folder="/home/tanlm/Downloads/lgg-mri-segmentation/kaggle_3m",
                                   image_label="/home/tanlm/Downloads/lgg-mri-segmentation/train.txt",
                                   is_train=True)

    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)

    for index, sample in enumerate(dataloader):
        print(sample['image'].size())
        for i in range(sample['image'].size()[0]):
            img = sample['image'].numpy()[i].transpose([1, 2, 0])
            mask = sample['mask'].numpy()[i]
            img *= (0.229, 0.224, 0.225)
            img += (0.485, 0.456, 0.406)
            img *= 255.0
            img = img.astype(np.uint8)
            cv2.imshow("img", img)
            cv2.imshow("mask", mask.astype(np.uint8))
            cv2.waitKey(0)
