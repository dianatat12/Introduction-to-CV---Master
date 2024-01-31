import numpy as np
import cv2


def find_text(queries, boxes):
    # detect bounding boxes in images and return the coordinates
    bboxes = {}
    i = 0

    for id, image in queries.items():
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([60, 60, 60], dtype=np.uint8)

        lower_white = np.array([170, 170, 170], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)

        # Create masks to identify pixels in the specified color ranges
        mask_black = cv2.inRange(image, lower_black, upper_black)
        mask_white = cv2.inRange(image, lower_white, upper_white)

        # Combine the black and white masks
        final_mask = cv2.bitwise_or(mask_black, mask_white)
        outside_mask = cv2.bitwise_not(final_mask)

        # Convert colors outside the threshold to a shade of gray
        image[outside_mask > 0] = [128, 128, 128]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((16, 8), np.uint8)
        toph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        di = cv2.morphologyEx(toph, cv2.MORPH_DILATE, kernel)
        cl = cv2.morphologyEx(di, cv2.MORPH_CLOSE, kernel)
        sorted_pixels = np.sort(np.unique(cl.ravel()))

        # Determine the threshold for the 10th lightest shade of white
        threshold = sorted_pixels[-15]
        defi = np.zeros_like(cl)

        # Set pixels with values greater than 214 to 255 in the new image
        defi[cl > threshold] = 255
        kernel = np.ones((15, 300), np.uint8)

        # ho dilato molt i despres faig la diferencia pq sajuntint els components
        end = cv2.morphologyEx(defi, cv2.MORPH_DILATE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(end)

        # Find the label with the largest connected component (excluding background label 0)
        largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        # Create a mask to keep only the largest component
        largest_component_mask = (labels == largest_component_label).astype(np.uint8)

        # Set all other components to black
        result_image = largest_component_mask * 255

        white_pixels = cv2.findNonZero(result_image)
        x, y, w, h = cv2.boundingRect(white_pixels)
        x = int(x + 80 / 2)
        w = int(w - 80)

        bboxes[id] = [x, y, x + w, y + h]

        image_with_box = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(image_with_box, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (255, 0, 0),
                      2)  # Red color

        i += 1

    return bboxes


def find_mask_w2(queries):
    masks = {}
    images_masked = {}
    for id, image in queries.items():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # equalize the image so the contrast between dark and light is improved
        eq = cv2.equalizeHist(gray)

        # perform a morfological closing to reduce black elements from background
        kernel = np.ones((2, 2), np.uint8)
        di = cv2.dilate(eq, kernel, iterations=1)
        kernel = np.ones((13, 13), np.uint8)
        ero = cv2.erode(di, kernel, iterations=2)

        norm = cv2.normalize(ero, None, 0, 255, cv2.NORM_MINMAX)

        kernel = np.ones((7, 13), np.uint8)
        gradient = cv2.morphologyEx(norm, cv2.MORPH_GRADIENT, kernel)

        _, binary_image = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours, filter using contour threshold area, and draw rectangle

        # inv = cv2.bitwise_not(binary_image)
        cnts = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        mask = np.zeros(binary_image.shape, dtype=np.uint8)
        counter = 0
        areas = {}

        for c in cnts:
            area = cv2.contourArea(c)  # shoelace formula for convex shapes
            areas[area] = c

        areass = dict(sorted(areas.items(), key=lambda item: item[0], reverse=True))

        # Get the two biggest values and their corresponding keys
        top_two = list(areass.items())[:2]
        top = list(areass.items())[:1]

        i = 0

        for area, c in top_two:
            if i == 0:
                x, y, w, h = cv2.boundingRect(c)  # get bounding box
                mask[y:y + h, x:x + w] = 255  # draw bounding box on mask

            if i > 0:
                # check that the second one is not much smaller
                x, y, w, h = cv2.boundingRect(c)
                if top[0][0] / area < 25:
                    mask[y:y + h, x:x + w] = 255
            i += 1

        # make sure that at least we have 1 mask
        all_black = np.all(mask == [0])

        if all_black:
            width, height = mask.size

            if height >= width:
                rectangle_width = width // 2
                rectangle_height = height // 2

                # Calculate the position of the white rectangle in the middle
                x1 = (width - rectangle_width) // 2
                y1 = (height - rectangle_height) // 2
                x2 = x1 + rectangle_width
                y2 = y1 + rectangle_height

                # Fill the white rectangle with white color
                mask[y1:y2, x1:x2] = [255]
            else:
                rectangle_width = width // 4  # Adjust as needed
                rectangle_height = height

                # Calculate the positions of the two vertical rectangles
                x1 = (width - 2 * rectangle_width) // 3
                x2 = x1 + rectangle_width + (width - 2 * rectangle_width) // 3
                y1 = 0
                y2 = height

                # Fill the first vertical rectangle with white color
                mask[y1:y2, x1:x1 + rectangle_width] = [255]

                # Fill the second vertical rectangle with white color
                mask[y1:y2, x2:x2 + rectangle_width] = [255]

        # save mask
        masks[id] = mask

        image[mask == 0] = [0, 0, 0]
        images_masked[id] = image

    return masks, images_masked
