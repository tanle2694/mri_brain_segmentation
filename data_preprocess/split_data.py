import cv2
import pathlib
import argparse
from random import shuffle
import os


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default="/home/tanlm/Downloads/data/kaggle_3m")
parser.add_argument("--output_link", default="/home/tanlm/Downloads/data/check_link")
args = parser.parse_args()

root_dir = pathlib.Path(args.root_dir)
output_link = pathlib.Path(args.output_link)
ratio = {"train": 0.6, "val": 0.2, "test": 0.2}

imgs = root_dir.rglob("*.tif")
images = []
masks = []
for link in imgs:
    if link.name.__contains__("mask.tif"):
        continue
    link_image = "{}/{}".format(link.parent.name, link.name)
    link_mask = "{}/{}_mask.tif".format(link.parent.name, link.name.split(".")[0])
    if not os.path.join(str(root_dir.absolute()), link_mask):
        continue
    masks.append(link_mask)
    images.append(link_image)
assert len(images) == len(masks)

indexs = list(range(len(images)))

shuffle(indexs)

train_val_split = int(len(images) * ratio['train'])
vali_test_split = train_val_split + int(ratio['val'] * len(images))

train_indexs = indexs[0: train_val_split]
vali_indexs = indexs[train_val_split: vali_test_split]
test_indexs = indexs[vali_test_split: ]
assert len(train_indexs + vali_indexs + test_indexs) == len(images)

train = '\n'.join(["{}\t{}".format(images[i], masks[i]) for i in train_indexs])
vali = '\n'.join(["{}\t{}".format(images[i], masks[i]) for i in vali_indexs])
test = '\n'.join(["{}\t{}".format(images[i], masks[i]) for i in test_indexs])

(output_link / "train.txt").write_text(train)
(output_link / "vali.txt").write_text(vali)
(output_link / "test.txt").write_text(test)