from math import floor
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy as sp


def make_cost_function(image):
    r = 0.6
    width = image.shape[1]
    height = image.shape[0]
    print("w, h", width, height)
    x0, x1 = 0, width - 1
    y0, y1 = int(floor((height - r*height)/2.0)), int(floor((height - r*height)/2.0 + r*height) - 1)
    canonical_image = np.zeros(image.shape, np.uint8)
    cv2.rectangle(canonical_image, (x0, y0), (x1, y1), (255, 255, 255), -1)

    def make_affine_matrix(transformation):
        M = np.asarray([[transformation[0], transformation[1], transformation[2]],
                        [transformation[3], transformation[4], transformation[5]]], np.float32)

        return M

    def cost_function(transformation):
        error_sum = 0.1
        m = make_affine_matrix(transformation)
        affine_canonical = cv2.warpAffine(canonical_image, m, (canonical_image.shape[0], canonical_image.shape[1]))
        # plt.imshow(affine_canonical, cmap='gray')
        # plt.xticks([]), plt.yticks([])
        # plt.show(block=True)

        for index, v in np.ndenumerate(image):
            if 0 <= index[0] < affine_canonical.shape[0] and 0 <= index[1] < affine_canonical.shape[1]:
                if affine_canonical[index[0], index[1], 0] != v:
                    error_sum += 1
            else:
                error_sum += 1

        return error_sum

    return cost_function
