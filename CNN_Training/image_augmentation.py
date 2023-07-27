import numpy as np
import random
from PIL import Image
from scipy import misc

def horzFlip(img):

    return aug_img

def vertFlip(img):

    return aug_img

def rotation(img, angle):

    return aug_img

def shift(img, xrange, yrange):
    aug_img = np.zeros(img.shape)
    yshift = np.int(img.shape[0] * random.uniform(-yrange, yrange))
    xshift = np.int(img.shape[1] * random.uniform(-xrange, xrange))

    print('xshift:', xshift, 'yshift:', yshift)

    # Negative yshift
    if yshift < 0:
        # Source Image Bounds
        src_ymin = yshift * -1
        src_ymax = img.shape[0]
        # Augmented Image Bounds
        fill_ymin = img.shape[0] + yshift
        fill_ymax = img.shape[0]
    # Positive yshift
    else:
        # Source Image Bounds
        src_ymin = 0
        src_ymax = img.shape[0] - yshift
        # Augmented Image Bounds
        fill_ymin = 0
        fill_ymax = yshift
    # Positive xshift
    if xshift < 0:
        # Source Image Bounds
        src_xmin = xshift * -1
        src_xmax = img.shape[1]
        # Augmented Image Bounds
        fill_xmin = img.shape[1] + xshift
        fill_xmax = img.shape[1]
    # Negative xshift
    else:
        # Source Image Bounds
        src_xmin = 0
        src_xmax = img.shape[1] - xshift
        # Augmented Image Bounds
        fill_xmin = 0
        fill_xmax = xshift

    # Fill augmented image with part of the original image
    aug_img[src_ymin:src_ymax, src_xmin:src_xmax, :] = img[src_ymin:src_ymax, src_xmin:src_xmax, :]
    # Fill the remaining gap with zeros or nearest
    # aug_img[fill_ymin:fill_ymax, fill_xmin:fill_xmax, :] = 0

    new_img = Image.fromarray(img)
    new_img.show()

    return aug_img

img = misc.imread('Dog.jpg')

print('Maximum potential yshift +/-:', np.int(img.shape[0] * .2),
      'Maximum potential xshift +/-:', np.int(img.shape[1] * .2))

# for x in range(10):
#     shift(img, .2, .2)

aug_img = shift(img, .2, .2)