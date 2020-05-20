from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import random
import numpy as np
import time
import torch.nn as nn
from logger.logger import get_logger
from dataloaders.dataloader import MRIBrainSegmentation
from modeling.deeplab import DeepLab
from utils.configure_parse import ConfigParser
import modeling.loss as loss
from trainer.trainer import Trainer
from torchvision import datasets, models, transforms


SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)



def main(args):
    config = ConfigParser(args)
    cfg = config.config
    logger = get_logger(config.log_dir, "train")

    train_dataset = MRIBrainSegmentation(root_folder=cfg['root_folder'], image_label=cfg['train_data'], is_train=True)
    vali_dataset = MRIBrainSegmentation(root_folder=cfg['root_folder'], image_label=cfg['validation_data'], is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["train_batch_size"], shuffle=True,
                                               num_workers=cfg["workers"], drop_last=True)

    vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=cfg["vali_batch_size"], shuffle=False,
                                              num_workers=cfg["workers"], drop_last=False)

    model = DeepLab(num_classes=cfg['nclass'],
                    backbone=cfg['backbone'],
                    output_stride=cfg['output_stride'],
                    sync_bn=cfg['sync_bn'],
                    freeze_bn=cfg['freeze_bn'])

    criterion = getattr(loss, 'cross_entropy')
    optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    metrics_name = ["accuracy"]
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, nb_epochs=config['epoch'],
                      valid_loader=vali_loader, logger=logger, log_dir=config.save_dir, metrics_name=metrics_name,
                      resume=config['resume'], save_dir=config.save_dir, device="cuda:1")
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="/home/tanlm/Downloads/covid_data/kfold/1/train.txt")
    parser.add_argument("--validation_data", default="/home/tanlm/Downloads/covid_data/kfold/1/val.txt")
    parser.add_argument("--root_folder", default="/home/tanlm/Downloads/covid_data/data")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out_stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--sync_bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze_bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--train_batch_size", default=32)
    parser.add_argument("--vali_batch_size", default=32)
    parser.add_argument("--nclass", default=2)
    parser.add_argument("--seed", default=1)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--weight_decay", default=4e-5)
    parser.add_argument("--max_iters", default=1000)
    parser.add_argument("--epoch", default=100)
    parser.add_argument("--resume", default="")
    parser.add_argument("--trainer_save_dir", default="/home/tanlm/Downloads/covid_data/save_dir")
    parser.add_argument("--exper_name", default="fold_1")
    args = parser.parse_args()
    main(args)