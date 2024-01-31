import cv2
import numpy as np
import math


def graylevel_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten().astype(int)  # Flatten the histogram and convert to integers
    return hist


def calculate_histogram(image, bins, bw=False, with_bbox=False, bbox=[]):
    hist = []
    if with_bbox:
        x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
        for channel_number in range(3):
            image[x:x + width, y:y + height, channel_number] = 0
            h = [elem[0] for elem in cv2.calcHist([image], [channel_number], None, [bins], [0, 255])]
            h[0] -= width * height
            hist.extend(h / sum(h))
    else:
        for channel_number in range(3):
            h = [elem[0] for elem in cv2.calcHist([image], [channel_number], None, [bins], [0, 255])]
            hist.extend(h / sum(h))

    # Add bw channel as well
    if bw:
        if with_bbox:
            x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

            image[x:x + width, y:y + height, channel_number] = 0
            h = [elem[0] for elem in cv2.calcHist([image], [0], None, [bins], [0, 255])]
            h[0] -= width * height
            hist.extend(h / sum(h))
        else:
            h = [elem[0] for elem in cv2.calcHist([image], [0], None, [bins], [0, 255])]
            hist.extend(h / sum(h))

    return hist


def divide_image(image, blocks_size):
    img = np.array(image)

    num_rows = img.shape[0]
    num_cols = img.shape[1]

    M = num_rows // (blocks_size // 2)
    N = num_cols // (blocks_size // 2)

    return [image[x:x + M, y:y + N] for x in range(0, num_rows, M) for y in range(0, num_cols, N)]


def calculate_histograms(images: dict, bins, path=None, with_blocks=False, bw=False, blocks_number=1, with_bboxes=False,
                         bboxes={}):
    histograms = {}

    for id, img in images.items():
        if with_blocks:
            if with_bboxes:
                histograms[id] = calculate_histogram_with_blocks(img, bins, blocks_number, with_bbox=True, bbox=bboxes[id])
            else:
                histograms[id] = calculate_histogram_with_blocks(img, bins, blocks_number)
        else:
            if with_bboxes:
                histograms[id] = calculate_histogram(img, bins, bw, with_bbox=True, bbox=bboxes[id])
            else:
                histograms[id] = calculate_histogram_with_blocks(img, bins, blocks_number)

    return histograms


def calculate_histogram_with_blocks(image, bins, blocks_number, with_bbox=False, bbox=[]):
    blocks = divide_image(image, blocks_number)

    hist = []
    for channel_number in range(3):
        for block in blocks:
            if not with_bbox:
                hist += [elem[0] for elem in cv2.calcHist([block], [channel_number], None, [bins], [0, 255])]
            else:
                x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

                image[x:x + width, y:y + height, channel_number] = 0
                h = [elem[0] for elem in cv2.calcHist([image], [channel_number], None, [bins], [0, 255])]
                h[0] -= width * height
                hist += [elem[0] for elem in cv2.calcHist([block], [channel_number], None, [bins], [0, 255])]

    return hist

def calculate_all_blocks(blocks, data, bins, with_bboxes=False, bboxes=[]):
    hist = {}
    for block in blocks:

        hist_per_image = []
        for key, image in data.items():
            all_blocks = divide_in_blocks_of_size(image, block)

            for b in all_blocks:
                if with_bboxes:
                    bbox = bboxes[key]
                else:
                    bbox = []
                h = calculate_histogram_per_block(b, bins, with_bbox=with_bboxes, bbox=bbox)
                hist_per_image.extend(h)

            hist[key] = hist_per_image

    return hist


# Taken create_blocks_array from Team8 - thank you!
def divide_in_blocks_of_size(image, blocks_size):
    # Set number of slices per axis
    axisSlice = int(math.sqrt(blocks_size))

    blocksArray = []
    # Split the image into vertical blocks
    split_h = np.array_split(image, axisSlice, axis = 0)
    for i in range(axisSlice):
        for j in range(axisSlice):
            # Split vertical blocks into square blocks
            split_hv = np.array_split(split_h[i], axisSlice, axis = 1)
            blocksArray.append(split_hv[j])
    return blocksArray


def calculate_histogram_per_block(block, bins,  with_bbox, bbox):
    if with_bbox:
        return calculate_histogram(block, bins, with_bbox=with_bbox, bbox=bbox)
    else:
        return calculate_histogram(block, bins)

def calculate_histogram_with_multiple_levels(images: dict, levels: list, bins, with_bboxes=False, bboxes={}):
    hist = calculate_histograms(images, bins, None, with_blocks=True, blocks_number=levels[0])

    # Calculate for other levels and concatenate to original
    for idx in range(1, len(levels)):
        block_size = levels[idx]
        if block_size > 1:
            hist_new = calculate_histograms(images, bins, None, with_blocks=True, blocks_number=block_size, with_bboxes=with_bboxes, bboxes=bboxes)
        else:
            hist_new = calculate_histograms(images, bins, None, with_blocks=False, with_bboxes=with_bboxes, bboxes=bboxes)

        for key_query in images.keys():
            extend_by = hist_new[key_query]
            hist[key_query].extend(extend_by)

    return hist
