from PIL import Image
from PIL import ImageOps
# from resizeimage import resizeimage
import cv2
import matplotlib.pyplot as plt
import numpy as np
import ntpath
import os

# img = Image.open('image.png').convert('LA')
# img.save('greyscale.png')

def get_file_base_name(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_file_base_name_without_extension(path):
    name_with_extension = get_file_base_name(path)
    name = name_with_extension.split('.')[0]

    return name

def rgb2grey(img):
    '''
    :param img: PIL image
    :return: grey image using 'LA'.
     CANNOT BE SAVED IN JPG FORMAT! SAVE AS PNG
    '''

    img_grey = img.convert('LA')

    return img_grey


def resize_image(img, size):
    '''
    Uses img which was read using PIL module!
    '''

    img_resized = img.resize(size)

    return img_resized

def flip_image(img):
    img_flipped = ImageOps.mirror(img)

    return img_flipped

def prepare_image_to_fit_learning_and_save(img_path, size, dst_path):
    if os.path.isdir(dst_path) is False:
        assert False, "dst_path should be a directory"

    if os.path.isdir(img_path) is True:
        assert False, "img_path should not be a directory"

    img = Image.open(img_path)

    file_name_without_extension = get_file_base_name_without_extension(img_path)

    grey_resized_file_name = file_name_without_extension + '_GR.png'
    grey_resized_flipped_file_name = file_name_without_extension + '_GRF.png'

    grey_resized_file_path = os.path.join(dst_path, grey_resized_file_name)

    grey_resized_flipped_file_path = os.path.join(dst_path, grey_resized_flipped_file_name)

    img_grey_resized = resize_image(rgb2grey(img), size)
    img_grey_resized.save(grey_resized_file_path)

    img_grey_resized_flipped = flip_image(img_grey_resized)
    img_grey_resized_flipped.save(grey_resized_flipped_file_path)

if __name__ == '__main__':

    img_path = '/Users/iliabenkovitch/Documents/Computer_Vision/git/tmp/test.jpg'

    size = np.array([800, 600])
    dst_path = '/Users/iliabenkovitch/Documents/Computer_Vision/git/tmp/'
    prepare_image_to_fit_learning_and_save(img_path, size, dst_path)
