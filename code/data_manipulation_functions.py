from PIL import Image
# from resizeimage import resizeimage
import cv2
import matplotlib.pyplot as plt
import numpy as np

# img = Image.open('image.png').convert('LA')
# img.save('greyscale.png')

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

def prepare_image_to_fit_learning_and_save(img, size, dst_path):
    img_grey_resized = resize_image(rgb2grey(img), size)
    img_grey_resized.save(dst_path)

if __name__ == '__main__':
    img = Image.open('/Users/iliabenkovitch/Documents/Computer_Vision/git/tmp/test.jpg')

    size = np.array([800, 600])
    dst_path = '/Users/iliabenkovitch/Documents/Computer_Vision/git/tmp/test_gery_rescaled.png'
    prepare_image_to_fit_learning_and_save(img, size, dst_path)

    # img_grey_resized = resize_image(rgb2grey(img), size)
    # plt.figure()
    # plt.imshow(img_grey_resized)

    # plt.show()