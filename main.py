import cv2
from components import find_components
from matplotlib import pyplot
import numpy as np
from PIL import Image


def write_bounding_box_images(cmps, image, original_image):
    # image_name = './data/FBs.tif'
    # original_image_name = './data/Im_crop.tif'
    #
    # image = cv2.imread(image_name)
    # original_image = cv2.imread(original_image_name)

    for cmp in cmps:
        cv2.rectangle(original_image, (cmp[1][0], cmp[1][1]), (cmp[1][2], cmp[1][3]), (255, 0, 0), 4)
        cv2.rectangle(image, (cmp[1][0], cmp[1][1]), (cmp[1][2], cmp[1][3]), (255, 0, 0), 4)

    cv2.imwrite('./output/bounding_box.png', original_image)
    cv2.imwrite('./output/bounding_box_blob.png', image)


def write_min_area_rect_images(cmps, image, original_image):
    idx = 0
    for cmp in cmps:
        non_zero = cv2.findNonZero(cmp[0])
        rect = cv2.minAreaRect(non_zero)
        rect = ((rect[0][0] + cmp[1][0], rect[0][1] + cmp[1][1]), (rect[1][0], rect[1][1]), rect[2])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], idx, (255, 0, 0), 3)
        cv2.drawContours(original_image, [box], idx, (255, 0, 0), 3)

    cv2.imwrite('./output/min_area_box.png', original_image)
    cv2.imwrite('./output/min_area_box_blob.png', image)


def main():
    image_name = './data/FBs.tif'
    original_image_name = './data/Im_crop.tif'
    dsm_image_name = './data/DSM_crop.tif'

    print("Reading images")
    image = cv2.imread(image_name)
    original_image = cv2.imread(original_image_name)
    dsm_image = np.asarray(Image.open(dsm_image_name)) # Use PILLOW to read 32bit tif


    print("Computing components")
    cmps = find_components(image)

    print("Writing bounding boxes")
    write_bounding_box_images(cmps, image.copy(), original_image.copy())

    print("Writing minimum area rectangles")
    write_min_area_rect_images(cmps, image.copy(), original_image.copy())


if __name__ == '__main__':
    main()
