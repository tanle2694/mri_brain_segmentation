from PIL import Image


def add_margin(pil_img, width_padding, height_padding, color):
    width, height = pil_img.size
    new_width = width + width_padding
    new_height = height + height_padding
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (0, 0))
    return result


img = Image.open("/home/tanlm/Downloads/data/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_2.tif")
img = add_margin(img, 50, 50, (255, 255, 255))
img.show()