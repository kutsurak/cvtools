import pickle

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import minimize

from components import find_components
from optimization import make_cost_function


def get_components(cmps, image):
    image_comps = []
    for cmp in cmps:
        image_comps.append(image[cmp[1][1]:cmp[1][3], cmp[1][0]:cmp[1][2]])

    return image_comps


def write_bounding_box_images(cmps, image, original_image):
    for cmp in cmps:
        cv2.rectangle(original_image, (cmp[1][0], cmp[1][1]), (cmp[1][2], cmp[1][3]), (255, 0, 0), 4)
        ret = cv2.rectangle(image, (cmp[1][0], cmp[1][1]), (cmp[1][2], cmp[1][3]), (255, 0, 0), 4)

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


def optimize_affine_transform(image_comps):
    i = 0
    results = []
    for img in image_comps:
        print("minimizing affine for component", i)
        cost_function = make_cost_function(img)
        aff0 = np.array([1, 0, 0, 1, 0, 0])
        res = minimize(cost_function, aff0, method='powell', options={'xtol':1e-8, 'disp': True})
        print(res.x)
        results.append(res)

        # plt.figure()
        # plt.imshow(img, cmap='gray')
        # plt.xticks([]), plt.yticks([])
        # plt.savefig('./output/blob_%i.png' % i)

        # m = make_affine_matrix(res.x)
        # canonical_image = make_canonical_images(img.shape[1], img.shape[0], 0.6, 0.4)
        # affine_image = cv2.warpAffine(canonical_image, m, (canonical_image.shape[1], canonical_image.shape[0]))

        # plt.figure()
        # plt.imshow(affine_image, cmap='gray')
        # plt.xticks([]), plt.yticks([])
        # plt.savefig('./output/affine_corrected_%i.png' % i)

        i += 1

    return results


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

    image_comps = get_components(cmps, image.copy())
    print('Total components', len(image_comps))

    res = optimize_affine_transform(image_comps)

    #save result to disk

    with open('./output/transformations.p', 'rb') as fl:
        pickle.dump(res, fl)

    # img = image_comps[5]
    # cost_function = make_cost_function(img)
    # aff0 = np.array([1, 0, 0, 1, 0, 0])
    # res = minimize(cost_function, aff0, method='powell', options={'xtol':1e-8, 'disp': True})
    # print(res.x)
    #
    # # m = make_affine_matrix(res.x)
    # height, width = img.shape[:2]
    # rh, rw = 0.4, 0.6
    # ps = [[int(floor((height - rh*height)/2.0)), int(floor((width - rw*width)/2.0))],
    #       [int(floor((height - rh*height)/2.0 + rh*height) - 1), int(floor((width - rw*width)/2.0))],
    #       [int(floor((height - rh*height)/2.0 + rh*height) - 1), int(floor((width - rw*width)/2.0 + rw*width) - 1)],
    #       [int(floor((height - rh*height)/2.0)), int(floor((width - rw*width)/2.0 + rw*width) - 1)]]
    # A = np.asarray([[res.x[0], res.x[1]], [res.x[2], res.x[3]]])
    # b = np.asarray([res.x[4], res.x[5]])
    # c = np.asarray(cmps[5][1][1], cmps[5][1][0])
    # afps = []
    # for p in ps:
    #     afps.append(np.dot(A, np.asarray(p)) + b + c)
    #
    # path = np.asarray(afps)
    # path = np.int0(path)
    # contour_image = cv2.drawContours(original_image.copy(), [path], 0, (255, 0, 0), 3)
    # plt.imshow(contour_image)
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    return res

if __name__ == '__main__':
    main()
