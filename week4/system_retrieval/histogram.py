from abc import abstractmethod

import cv2
import numpy as np


class Histogram:
    def __init__(self, bins):
        self.bins = bins

    @abstractmethod
    def calculate_histogram(self, img, with_bbox=False, text_box=None):
        pass


class GrayscaleHistogram(Histogram):
    def __init__(self, bins):
        self.bins = bins

    def calculate_histogram(self, img, with_bbox=False, text_box=None):
        hist = cv2.calcHist([img], [0], None, [self.bins], [0, 256])
        hist = hist.flatten().astype(int)  # Flatten the histogram and convert to integers
        return hist


class ThreeChannelHistogram(Histogram):
    def __init__(self, bins):
        self.bins = bins

    def calculate_histogram(self, img, with_bbox=False, text_box=None):
        hist = []
        for channel_number in range(3):
            h = [elem[0] for elem in cv2.calcHist([img], [channel_number], None, [self.bins], [0, 255])]
        hist.extend(h / sum(h))

        return hist


class HistogramWithBlocks(ThreeChannelHistogram):
    def __init__(self, bins, block_size):
        self.bins = bins
        self.block_size = block_size

    # Taken from the internet
    def calculate_histogram(self, img, with_bbox=False, text_bbox=None):
        sizeX = img.shape[1] #w
        sizeY = img.shape[0] #h

        SF_W = 1024 / sizeX
        SF_H = 1024 / sizeY
        

        resized_img = cv2.resize(img, (1024, 1024))
        if with_bbox and (text_bbox is not None):
            tlx_bbox = int(text_bbox[0]* SF_W)
            tly_bbox = int(text_bbox[1]* SF_H)
            brx_bbox = int(text_bbox[2]* SF_W + text_bbox[0]* SF_W)
            bry_bbox = int(text_bbox[3]* SF_H + text_bbox[1]* SF_H)

        sizeX = resized_img.shape[1] #w
        sizeY = resized_img.shape[0]

        hist = []
        for i in range(0,self.block_size):
            for j in range(0, self.block_size):
                # block
                img_block = resized_img[int(i*sizeY/self.block_size):int(i*sizeY/self.block_size) + int(sizeY/self.block_size) ,int(j*sizeX/self.block_size):int(j*sizeX/self.block_size) + int(sizeX/self.block_size)]

                if not with_bbox or (text_bbox is None):
                    hist_partial = ThreeChannelHistogram(bins=self.bins).calculate_histogram(img_block, self.bins)

                # If there's a text bounding box ignore the pixels inside it
                else:
                    tlx = tlx_bbox-int(j*sizeX/self.block_size)
                    tly = tly_bbox-int(i*sizeY/self.block_size)
                    brx = brx_bbox-int(j*sizeX/self.block_size)
                    bry = bry_bbox-int(i*sizeY/self.block_size)

                    valid_pixels = []

                    for x in range(img_block.shape[1]-1):
                        for y in range(img_block.shape[0]-1):
                            if not (tlx < x < brx and  tly < y < bry):
                                valid_pixels.append(img_block[y,x,:])

                    valid_pixels = np.asarray(valid_pixels)



                    if valid_pixels.size!=0:
                        valid_pixels_matrix = np.reshape(valid_pixels,(valid_pixels.shape[0],1,-1))
                        hist_partial = ThreeChannelHistogram(self.bins).calculate_histogram(valid_pixels_matrix)

                hist.extend(hist_partial)

        return hist


class MultiLevelHistogram():
    def __init__(self, bins, levels):
        self.levels = levels
        self.bins = bins

    def calculate_histogram(self, img, with_bbox=False, text_box=None):
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = []

        for bl in self.levels:
            hist_calculator = HistogramWithBlocks(self.bins, block_size=bl)
            hist_partial = hist_calculator.calculate_histogram(lab_image, with_bbox, text_box)
            hist.extend(hist_partial)

        return hist


