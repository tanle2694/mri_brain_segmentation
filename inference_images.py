from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
from dataloaders.dataloader import MRIBrainSegmentation
import cv2
import os
from modeling.deeplab import DeepLab
import torch.nn.functional as F

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args):
    vali_dataset = MRIBrainSegmentation(root_folder=args.root_folder,
                                        image_label=args.data_label,
                                        is_train=False)
    vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=16, shuffle=False,
                                              num_workers=4, drop_last=False)

    # Init and load model
    model = DeepLab(num_classes=1,
                    backbone='resnet',
                    output_stride=8,
                    sync_bn=None,
                    freeze_bn=False)

    checkpoint = torch.load(args.checkpoint)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(vali_loader):
            print(i)
            data = sample['image']
            target = sample['mask']
            data, target = data.to(device), target.to(device)
            output = model(data)

            target = target.data.cpu().numpy()
            data = data.data.cpu().numpy()
            output = output.data.cpu().numpy()
            pred = np.zeros_like(output)
            pred[output > 0.5] = 1
            pred = pred[:, 0]
            for j in range(len(target)):
                output_image = pred[j] * 255
                target_image = target[j] * 255

                cv2.imwrite("{}/{:06d}_{:06d}_predict.png".format(args.output_folder, i, j), output_image.astype(np.uint8))
                cv2.imwrite("{}/{:06d}_{:06d}_target.png".format(args.output_folder, i, j), target_image.astype(np.uint8))
                img = data[j].transpose([1, 2, 0])
                img *= (0.229, 0.224, 0.225)
                img += (0.485, 0.456, 0.406)
                img *= 255.0
                cv2.imwrite(
                    "{}}/{:06d}_{:06d}_origin.png".format(args.output_folder,
                        i, j), img.astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_label", default="/home/tanlm/Downloads/data/vali.txt")
    parser.add_argument("--root_folder", default="/home/tanlm/Downloads/data")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--vali_batch_size", default=16)
    parser.add_argument("--seed", default=1)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output_folder", default="data/result.txt")
    args = parser.parse_args()
    main(args)