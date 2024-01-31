import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
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

def graylevel_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten().astype(int)  # Flatten the histogram and convert to integers
    return hist

def calculate_histogram(image, bins, bw):
    hist = []
    for channel_number in range(3):
        h = [elem[0] for elem in cv2.calcHist([image], [channel_number], None, [bins], [0, 255])]
        hist.extend(h / sum(h))

    # Add bw channel as well
    if bw:
        h = [elem[0] for elem in cv2.calcHist([image], [0], None, [bins], [0, 255])]
        hist.extend(h / sum(h))
    return hist


def calculate_histograms(images: dict, bins, path=None, with_blocks=False, bw=False, blocks_number=1):
    histograms = {}

    for id, img in images.items():
        if with_blocks:
            histograms[id] = calculate_histogram_with_blocks(img, bins, blocks_number)
        else:
            histograms[id] = calculate_histogram(img, bins, bw)

    return histograms


def calculate_histogram_with_blocks(image, bins, blocks_number):
    hist = []
    blocks = divide_image(image, blocks_number)

    for channel_number in range (3):
        # TODO: H takes values only up to 179
        for block in blocks:
            hist += [elem[0] for elem in cv2.calcHist([block], [channel_number], None, [48], [0, 255])]

    return hist


def chi2_distance(A, B):
    chi = 0
    for (a, b) in zip(A, B):
        if a+b != 0:
            chi += ((a - b) ** 2) / (a + b)

    return chi*0.5


def intersection_per_query(hist_query, hist_bbdd: dict, k):
    result_query = {}
    for key, value in hist_bbdd.items():
        result_query[key] = histogram_intersection(hist_query, value)

    best_k = dict(sorted(result_query.items(), key=lambda item: item[1], reverse=True)[:k])
    return [extract_number_from_string(key) for key in best_k.keys()]


def histogram_intersection(hist1, hist2):
    return np.sum(np.minimum(hist1, hist2))


# https://datagy.io/manhattan-distance-python/
# https://www.statology.org/manhattan-distance-python/
def l1_distance(hist1, hist2):
    return np.sum(np.abs(hist1 - hist2))

def hellinger(p, q):
    return sum([(math.sqrt(t[0])-math.sqrt(t[1]))*(math.sqrt(t[0])-math.sqrt(t[1])) \
                for t in zip(p,q)])/math.sqrt(2.)


def show_differences(hist1, hist2, img1, img2, bins):
    plot_histograms(hist1, hist2, bins)
    plot_images(img1, img2)

    return histogram_intersection(hist1, hist2)


def plot_histograms(hist1, hist2, bins):
    x = np.linspace(0, len(hist1) - 1, len(hist1))
    plt.bar(x, hist1)
    plt.bar(x, hist2)
    plt.show()


def plot_images(img1, img2):
    plt.subplot(2, 1, 1)
    plt.imshow(img1)
    plt.title('image1')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 1, 2)
    plt.imshow(img2)
    plt.title('image2')
    plt.xticks([])
    plt.yticks([])

def divide_image(image, blocks_number):
    img = np.array(image)

    num_rows = img.shape[0]
    num_cols = img.shape[1]

    M = num_rows//(blocks_number // 2)
    N = num_cols//(blocks_number // 2)

    return [image[x:x+M,y:y+N] for x in range(0,num_rows,M) for y in range(0,num_cols,N)]


# Written by chatGPT
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


def import_images_gt(path):
    images = {}

    for name in os.listdir(path):
        if name.endswith(".png"):
            image = cv2.imread(os.path.join(path, name))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            images[name[:-4]] = gray

    return images

def find_mask_2(queries):
    masks = {}
    for id, image in queries.items():
        l = []

        col_ini = list(image[:, :5, :].reshape((-1, 3)))
        l.extend(col_ini)
        col_end = list(image[:, -5:, :].reshape((-1, 3)))
        l.extend(col_end)
        row_ini = list(image[:5, :, :].reshape((-1, 3)))
        l.extend(row_ini)
        row_end = list(image[-5:, :, :].reshape((-1, 3)))
        l.extend(row_end)

        columns = list(map(list, zip(*l)))
        max_values = []
        min_values = []

        # Iterate through each column
        for column in columns:
            max_val = max(column)
            min_val = min(column)
            max_values.append(max_val)
            min_values.append(min_val)

        print(min_values, max_values)
        mask = np.logical_and.reduce((image[:, :, 0] >= min_values[0],image[:, :, 1] >= min_values[1],image[:, :, 2] >= min_values[2],image[:, :, 0] <= max_values[0],image[:, :, 1] <= max_values[1],image[:, :, 2] <= max_values[2] ))

        print(image[0][0])
        image[mask] = [0, 0, 0]  # RGB color for black
        cv2.imshow(image)
        print(mask)

def find_mask(queries):
    masks = {}
    images_masked = {}
    for id, image in queries.items():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # equalize the image so the contrast between dark and light is improved
        eq = cv2.equalizeHist(gray)

        # perform a morfological closing to reduce black elements from background
        kernel = np.ones((2,2), np.uint8)
        di = cv2.dilate(eq, kernel, iterations=1)
        kernel = np.ones((13,13), np.uint8)
        ero = cv2.erode(di, kernel, iterations = 2)

        norm = cv2.normalize(ero, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(norm,(11,11), sigmaX=0)

        # Apply Otsu's thresholding to get a binnary picture
        _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # find components inside painting and remove them
        num_labels, labels = cv2.connectedComponents(binary_image)

        # Iterate through labels
        for label in range(2, num_labels):  # Start from 1 to skip the background label (0)
            # Create a mask for the current component
            component_mask = np.uint8(labels == label)
            component_mask[component_mask==1] = 255
            binary_image[component_mask == 255] = 0
        
        # invert image (black <--> white)
        inv = cv2.bitwise_not(binary_image)

        # save mask
        masks[id] = inv
        mask = inv == 0

        #save masked image
        image[mask] = [0, 0, 0]
        images_masked[id] = image

    return masks, images_masked


def eval(gt, masks):
    p = 0
    r = 0
    f = 0
    for (k1, v1), (k2, v2) in zip(gt.items(), masks.items()):
        TP = np.sum(np.logical_and(v1 == 255, v2 == 255))
        FP = np.sum(np.logical_and(v1 == 0, v2 == 255))
        TN = np.sum(np.logical_and(v1 == 0, v2 == 0))
        FN = np.sum(np.logical_and(v1 == 255, v2 == 0))

        # Calculate Precision, Recall, and F1 Score
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
        p += precision
        r += recall
        f += f1_score
    return p / 30, r / 30, f / 30

