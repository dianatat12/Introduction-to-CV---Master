import numpy as np
import cv2

class TextureDescriptor:
    def calculate(self, img):
        pass


class DctDescriptor(TextureDescriptor):
    def __init__(self, num_blocks, N):
        self.num_blocks = num_blocks
        self.N = N


    def divide_image(self, img, num_blocks):
        # Get the dimensions of the image
        height, width = img.shape

        # Calculate the size of each block
        block_width = width // num_blocks
        block_height = height // num_blocks

        # Create a list to store the divided blocks
        blocks = []

        for j in range(num_blocks):
            for i in range(num_blocks):
                # Calculate the coordinates of the top-left and bottom-right corners of the block
                x1 = i * block_width
                y1 = j * block_height
                x2 = (i + 1) * block_width
                y2 = (j + 1) * block_height

                # Crop the block from the img
                block = img[y1:y2, x1:x2]

                # Add the block to the list
                blocks.append(block)

        return blocks

    def zigzag(self, input):
        # initializing the variables
        #----------------------------------
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        vmax = input.shape[0]
        hmax = input.shape[1]

        i = 0

        output = np.zeros(( vmax * hmax))
        #----------------------------------

        while ((v < vmax) and (h < hmax)):

            if ((h + v) % 2) == 0:                 # going up

                if (v == vmin):
                    output[i] = input[v, h]        # if we got to the first line

                    if (h == hmax):
                        v = v + 1
                    else:
                        h = h + 1

                    i = i + 1

                elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                    output[i] = input[v, h]
                    v = v + 1
                    i = i + 1

                elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                    output[i] = input[v, h]
                    v = v - 1
                    h = h + 1
                    i = i + 1


            else: # going down
                if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                    output[i] = input[v, h]
                    h = h + 1
                    i = i + 1

                elif (h == hmin):                  # if we got to the first column
                    output[i] = input[v, h]

                    if (v == vmax -1):
                        h = h + 1
                    else:
                        v = v + 1

                    i = i + 1

                elif ((v < vmax -1) and (h > hmin)):     # all other cases
                    output[i] = input[v, h]
                    v = v + 1
                    h = h - 1
                    i = i + 1


            if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
                output[i] = input[v, h]
                break

        return output

    def compute_DCT_blocks(self, image, num_blocks, N):
        # Divide the image in blocks
        blocks = self.divide_image(image, num_blocks)

        # Compute the DCT descriptor for each block
        img_DCT = []
        for img in blocks:
            img_float = np.float32(img)/255.0      # float conversion/scale
            DCT = cv2.dct(img_float)               # the dct
            img_DCT.append(np.uint8(DCT)*255.0)    # convert back

        # Scan the image in zig-zag
        img_zigzag = []
        for i in range(num_blocks*num_blocks):
            img_zigzag.append(self.zigzag(img_DCT[i]))

        # Keep the first N coefficients
        block_feature_vector = []
        for i in range(num_blocks*num_blocks):
            for j in range(N):
                block_feature_vector.append(img_zigzag[i][j])

        return block_feature_vector

    def calculate(self, img):
        return self.compute_DCT_blocks(img, self.num_blocks, self.N)
