import cv2
import numpy as np

def find_mask_2_w1(queries):
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

def find_mask_w1(queries):
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


