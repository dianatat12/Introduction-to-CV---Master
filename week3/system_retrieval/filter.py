from abc import abstractmethod
import cv2
import numpy as np


class Filter:
    def __init__(self):
        pass

    @abstractmethod
    def filter(self, img):
        pass


class MedianFilter(Filter):
    def __init__(self, neighborhood_size: int):
        super().__init__()
        self.neighborhood_size = neighborhood_size

    def filter(self, img):
        dims = img.shape
        img_filtered = img.copy()
        half_size = self.neighborhood_size // 2

        for i in range(half_size, dims[0] - half_size):
            for j in range(half_size, dims[1] - half_size):
                for c in range(dims[2]):  # Loop through channels (BGR)
                    v = img[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1, c]
                    v = np.sort(v, axis=None)
                    img_filtered[i, j, c] = v[v.size // 2]

        return img_filtered


class AverageFilter(Filter):
    def __init__(self, ksize: int):
        super().__init__()
        self.ksize = ksize

    def filter(self, img):
        
        return cv2.blur(img, (self.ksize,self.ksize))

class MaxRankFilter(Filter):
    def __init__(self, size_input: int, output_rank: int):
        super().__init__()
        self.size_input = size_input
        self.output_rank = output_rank

    def filter(self, img):
        # Split the BGR image into color channels
        blue_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        red_channel = img[:, :, 2]

        # Initialize an empty output image
        output_img = np.zeros_like(img)

        # Apply the Max Rank Filter to each color channel
        for channel in (blue_channel, green_channel, red_channel):
            dims = channel.shape

            img_rank = channel.copy()

            for i in range(int(self.size_input / 2), dims[0] - int(self.size_input / 2)):
                for j in range(int(self.size_input / 2), dims[1] - int(self.size_input / 2)):
                    V = channel[i - int(self.size_input / 2):i + int(self.size_input / 2) + 1,
                        j - int(self.size_input / 2):j + int(self.size_input / 2) + 1]
                    V = np.sort(V, axis=None)
                    img_rank[i, j] = V[self.output_rank]

            # Copy the filtered channel to the corresponding channel in the output image
            output_img[:, :, 0] = blue_channel
            output_img[:, :, 1] = green_channel
            output_img[:, :, 2] = red_channel
        return output_img


class WeightedMaxRankFilter(Filter):
    def __init__(self, size_input: int, output_rank: int):
        super().__init__()
        self.size_input = size_input
        self.output_rank = output_rank

    def filter(self, img):
        # Split the BGR image into color channels
        blue_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        red_channel = img[:, :, 2]

        # Initialize empty arrays for the filtered channels
        filtered_blue = np.zeros_like(blue_channel)
        filtered_green = np.zeros_like(green_channel)
        filtered_red = np.zeros_like(red_channel)

        # Apply the Max Rank Filter to each color channel
        for channel, filtered_channel in zip((blue_channel, green_channel, red_channel),
                                             (filtered_blue, filtered_green, filtered_red)):
            dims = channel.shape

            for i in range(int(self.size_input / 2), dims[0] - int(self.size_input / 2)):
                for j in range(int(self.size_input / 2), dims[1] - int(self.size_input / 2)):
                    V = channel[i - int(self.size_input / 2):i + int(self.size_input / 2) + 1,
                        j - int(self.size_input / 2):j + int(self.size_input / 2) + 1]
                    V = np.sort(V, axis=None)
                    filtered_channel[i, j] = V[self.output_rank]

        # Recombine the filtered channels to create the output image
        output_img = np.stack((filtered_blue, filtered_green, filtered_red), axis=-1)

        return output_img

# Extend Filter to use
class LowPassFilter():
    def filter(self, img):
        # Perform a Fourier Transform on the image
        f_transform = np.fft.fft2(img)
        f_transform_shifted = np.fft.fftshift(f_transform)

        # Create a mask for the low-pass filter
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        d = 70

        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - d:crow + d, ccol - d:ccol + d] = 1

        # Apply the mask to the Fourier Transform
        f_transform_shifted_filtered = f_transform_shifted * mask

        # Perform an inverse Fourier Transform
        f_transform_filtered = np.fft.ifftshift(f_transform_shifted_filtered)
        img_filtered = np.fft.ifft2(f_transform_filtered)
        img_filtered = np.abs(img_filtered)

        return img_filtered

# Extend Filter to use
class HighPassFilter():
    def filter(self, img):
        # Perform a Fourier Transform on the image
        f_transform = np.fft.fft2(img)
        f_transform_shifted = np.fft.fftshift(f_transform)

        # Create a mask for the high-pass filter
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        d = 10

        mask = np.ones((rows, cols), np.uint8)
        mask[crow - d:crow + d, ccol - d:ccol + d] = 0

        # Apply the mask to the Fourier Transform
        f_transform_shifted_filtered = f_transform_shifted * mask

        # Perform an inverse Fourier Transform
        f_transform_filtered = np.fft.ifftshift(f_transform_shifted_filtered)
        img_filtered = np.fft.ifft2(f_transform_filtered)
        img_filtered = np.abs(img_filtered)

        return img_filtered


# Extend Filter to use
class GaussianFilter():
    def filter(self, img):
        ksize = (3, 3)
        sigma = 10

        return cv2.GaussianBlur(img, ksize, sigma)


# Extend Filter to use
class BandpassFilter():
    def filter(self, img):
        # Perform a Fourier Transform on the image
        f_transform = np.fft.fft2(img)
        f_transform_shifted = np.fft.fftshift(f_transform)

        # Define the bandwidth (range of frequencies to retain)
        low_cutoff = 30
        high_cutoff = 80

        # Create a mask for the band-pass filter
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - high_cutoff:crow + high_cutoff, ccol - high_cutoff:ccol + high_cutoff] = 1
        mask[crow - low_cutoff:crow + low_cutoff, ccol - low_cutoff:ccol + low_cutoff] = 0

        # Apply the mask to the Fourier Transform
        f_transform_shifted_filtered = f_transform_shifted * mask

        # Perform an inverse Fourier Transform
        f_transform_filtered = np.fft.ifftshift(f_transform_shifted_filtered)
        img_filtered = np.fft.ifft2(f_transform_filtered)
        img_filtered = np.abs(img_filtered)

        return img_filtered
