import numpy as np


def evaluate(gt, masks):
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


def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
