from skimage import measure
import cv2
import numpy as np


def show_image(img):
    cv2.imshow('result', img), cv2.waitKey(0)


if __name__ == '__main__':
    image = cv2.imread('./data/FBs.tif')
    img = cv2.split(image)[0]
    show_image(img)

    labels = measure.label(img, background=0)
    print("Number of labels = ", np.max(labels))
    label_number = 0
    while True:
        temp = np.uint8(labels==label_number) * 255
        if not cv2.countNonZero(temp):
            break
        t = cv2.findNonZero(temp).T
        (min_i, max_i, min_j, max_j) = (np.min(t[0]), np.max(t[0]), np.min(t[1]), np.max(t[1]))

        show_image(temp[min_j:max_j, min_i:max_i])
        label_number += 1

    cv2.destroyAllWindows()
