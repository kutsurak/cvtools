from math import floor
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy as sp


def make_canonical_images(width, height, rw, rh):
    x0, x1 = int(floor((width - rw*width)/2.0)), int(floor((width - rw*width)/2.0 + rw*width) - 1)
    # x0, x1 = 0, width - 1
    y0, y1 = int(floor((height - rh*height)/2.0)), int(floor((height - rh*height)/2.0 + rh*height) - 1)
    canonical_image = np.zeros((height, width, 3), np.uint8)
    cimage = cv2.rectangle(canonical_image, (x0, y0), (x1, y1), (255, 255, 255), -1)

    return cimage


def make_affine_matrix(transformation):
    m = np.asarray([[transformation[0], transformation[1], transformation[4]],
                    [transformation[2], transformation[3], transformation[5]]],
                   np.float32)

    return m


def make_cost_function(image):
    canonical_image = make_canonical_images(image.shape[1], image.shape[0], 0.6, 0.4)

    def cost_function(transformation):
        error_sum = 0
        m = make_affine_matrix(transformation)

        affine_canonical = cv2.warpAffine(canonical_image, m, (canonical_image.shape[1], canonical_image.shape[0]))
        # plt.imshow(affine_canonical, cmap='gray')
        # plt.xticks([]), plt.yticks([])
        # plt.show(block=True)

        for index, v in np.ndenumerate(image):
            if 0 <= index[0] < affine_canonical.shape[1] and 0 <= index[1] < affine_canonical.shape[0]:
                if affine_canonical[index[0], index[1], 0] != v:
                    error_sum += 1
            else:
                error_sum += 1

        return error_sum

    return cost_function
