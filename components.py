from skimage import measure
import cv2
import numpy as np


def show_image(img):
    cv2.imshow('result', img), cv2.waitKey(0)


def find_components(image):
    """
    Given a binary image this function returns a list of images containing the connected components
    """

    img = cv2.split(image)[0]

    labels = measure.label(img, background=0)
    # print("Number of labels = ", np.max(labels))
    label_number = 0
    components = []
    while True:
        temp = np.uint8(labels == label_number) * 255
        if not cv2.countNonZero(temp):
            break

        t = cv2.findNonZero(temp).T
        (min_i, max_i, min_j, max_j) = (np.min(t[0]), np.max(t[0]), np.min(t[1]), np.max(t[1]))
        components.append(temp[min_j:max_j, min_i:max_i])
        label_number += 1

    return components
