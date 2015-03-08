import cv2
from components import find_components
from matplotlib import pyplot
import numpy as np


def write_output_images(cmps):
    image_name = './data/FBs.tif'
    original_image_name = './data/Im_crop.tif'

    image = cv2.imread(image_name)
    original_image = cv2.imread(original_image_name)

    for cmp in cmps:
        cv2.rectangle(original_image, (cmp[1][0], cmp[1][1]), (cmp[1][2], cmp[1][3]), (255, 0, 0), 4)
        cv2.rectangle(image, (cmp[1][0], cmp[1][1]), (cmp[1][2], cmp[1][3]), (255, 0, 0), 4)


    # pyplot.imshow(original_image)
    # pyplot.xticks([]), pyplot.yticks([])
    # pyplot.show()

    pyplot.imshow(image)
    pyplot.show()

    cv2.imwrite('./output/bounding_box.png', original_image)
    cv2.imwrite('./output/bounding_box_blob.png', image)


def main():
    image_name = './data/FBs.tif'
    original_image_name = './data/Im_crop.tif'


    image = cv2.imread(image_name)
    original_image = cv2.imread(original_image_name)


    cmps = find_components(image)

    idx = 0
    for cmp in cmps:
        non_zero = cv2.findNonZero(cmp[0])
        rect = cv2.minAreaRect(non_zero)
        rect = ((rect[0][0] + cmp[1][0], rect[0][1] + cmp[1][1]), (rect[1][0], rect[1][1]), rect[2])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], idx, (255, 0, 0), 3)
        cv2.drawContours(original_image, [box], idx, (255, 0, 0), 3)
        #print(box)
        #cv2.rectangle(image, )

    # pyplot.imshow(image)
    # pyplot.xticks([]), pyplot.yticks([])
    # pyplot.show()
    #
    # pyplot.imshow(original_image)
    # pyplot.show()

    cv2.imwrite('./output/min_area_box.png', original_image)
    cv2.imwrite('./output/min_area_box_blob.png', image)

    write_output_images(cmps)


if __name__ == '__main__':
    main()
