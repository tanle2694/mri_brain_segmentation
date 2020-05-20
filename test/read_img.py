import cv2
import pathlib
import numpy as np

root_dir = pathlib.Path("/home/tanlm/Downloads/lgg-mri-segmentation/kaggle_3m")

imgs = root_dir.rglob("*.tif")
for link in imgs:
    print(str(link.absolute()))
    if str(link.absolute()).__contains__("mask"):
        continue
    img = cv2.imread(str(link.absolute()))
    img_resize = cv2.resize(img, (int(img.shape[1] * 1.2), int(img.shape[0] * 1.2)))
    cv2.imshow("df", img)
    x = img_resize.shape[0] // 4
    print(x)
    cv2.imshow("resize", img_resize[x: 3*x, x: 3*x])
    cv2.waitKey(0)
