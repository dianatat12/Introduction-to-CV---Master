import os
import cv2
import numpy as np
import math

def import_images(path):
    images = {}

    for name in os.listdir(path):
        if name.endswith(".jpg"):
            image = cv2.imread(os.path.join(path, name))
            image = cv2.imread(os.path.join(path, name))

            images[name[:-4]] = image

    return images


def resize_images(images: dict, size):
    for idx, img in images.items():
        images[idx] = cv2.resize(img, size)
    return images



def convert_color_space(images: dict, code):
    new = {}
    for idx, imag in images.items():
        new[idx] = cv2.cvtColor(imag, code)
    return new


def import_images_gt(path):
    images = {}

    for name in os.listdir(path):
        if name.endswith(".png"):
            image = cv2.imread(os.path.join(path, name))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            images[name[:-4]] = gray

    return images

