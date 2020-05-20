import cv2
import os
import numpy as np


def get_image_and_merge(input_folder, file_name):
    img_background = np.zeros((300, 300, 3), dtype=np.uint8)
    start_x = 0
    start_y = 0
    nb = 0
    for img_dir in os.listdir(input_folder):
        if start_y >=300:
            start_x += 60
            start_y=0
            if start_x >=300:
                break
        print(nb)
        print(start_x, start_y)
        nb += 1
        img = cv2.imread(os.path.join(input_folder, img_dir))
        img = cv2.resize(img, (60, 60))
        img_background[start_x: start_x + 60, start_y: start_y + 60, :] = img
        start_y += 60
    cv2.imwrite(file_name, img_background.astype(np.uint8))

get_image_and_merge("/home/tanlm/Downloads/covid_data/data/Nomal_Disease_CT", "Nomal_Disease_CT.png")
get_image_and_merge("/home/tanlm/Downloads/covid_data/data/COVID_Positive_CT", "COVID_Positive_CT.png")
get_image_and_merge("/home/tanlm/Downloads/covid_data/data/No_Finding_CT", "No_Finding_CT.png")
