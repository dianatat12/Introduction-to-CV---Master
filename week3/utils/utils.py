import os
import cv2
import numpy as np
import re

def import_images(path):
    images = {}

    for name in os.listdir(path):
        if name.endswith(".jpg"):
            image = cv2.imread(os.path.join(path, name))

            images[name[:-4]] = image

    return images

def import_text(path):
    text = {}
    for name in os.listdir(path):
        if name.endswith(".txt"):
            with open(path + "/" + name, 'r', encoding='latin-1') as file:
                for line in file:
                    split_strings = line[2:-2].split("', '")
                    clean = [elem.replace("'", "") for elem in split_strings]

            text[name[:-4]] = clean

    return text


def extract_number_from_string(input_string):
    # Define a regular expression pattern to match the number part.
    pattern = r'\d+'

    # Use re.search to find the first match in the input string.
    match = re.search(pattern, input_string)

    # Check if a match was found.
    if match:
        # Extract the matched number as a string.
        number_str = match.group(0)

        # Convert the extracted string to an integer.
        number = int(number_str)

        return number
    else:
        # Return None if no match was found.
        return None

def resize_images(images: dict, size):
    for idx, img in images.items():
        images[idx] = cv2.resize(img, size)
    return images


def import_text(path):
    text = {}
    for name in os.listdir(path):
        if name.endswith(".txt"):
            with open(path + "/" + name, 'r', encoding='latin-1') as file:
                for line in file:
                    split_strings = line[2:-2].split("', '")
                    clean = [elem.replace("'", "") for elem in split_strings]

            text[name[:-4]] = clean

    return text


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


# Taken from Team 5 - thank you!
def erode(img):
    h, w = img.shape
    hper = 0.007
    wper = 0.007
    hker = int(h * hper)
    wker = int(w * wper)

    kernel = np.ones((hker, wker), np.uint8)
    erode = cv2.erode(img, kernel, iterations=5)

    return erode


# Taken from Team 5 - thank you!
def dilate(img):
    h, w = img.shape
    hper = 0.005
    wper = 0.005
    hker = int(h * hper)
    wker = int(w * wper)

    kernel = np.ones((hker, wker), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=5)

    return dilate


# Taken from Team 5 - thank you!
def separate_image(image: np.ndarray, mask: np.ndarray) -> list:
    cropped_images = []
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(gray_masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(
        contours, key=lambda x: cv2.contourArea(x, True), reverse=False
    )[:2]

    def get_x_coordinate(contour):
        x, _, _, _ = cv2.boundingRect(contour)
        return x

    contours = sorted(contours, key=get_x_coordinate, reverse=False)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w > image.shape[1] * 0.1 and h > image.shape[0] * 0.1:
            cropped_images.append(masked_image[y:y + h, x:x + w])

    return cropped_images


# Taken from Team 5 - thank you!
def remove_background(mask: np.ndarray, image: np.ndarray) -> list:
    if mask is None:
        return np.ones(image.shape[:2], dtype=np.uint8) * 255

    return separate_image(image, mask)


def intersection_per_query(hist_query, hist_bbdd: dict, k):
    result_query = {}
    for key, value in hist_bbdd.items():
        result_query[key] = histogram_intersection(hist_query, value)

    best_k = dict(sorted(result_query.items(), key=lambda item: item[1], reverse=True)[:k])
    return [extract_number_from_string(key) for key in best_k.keys()]

def histogram_intersection(hist1, hist2):
    return np.sum(np.minimum(hist1, hist2))
