# The source of this script is from:
# https://github.com/TejasNaikk/Image-Alignment-and-Panoramas/blob/master/main.py

import cv2
import numpy as np
import math

'''
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching
Usage example:
    im = cv2.imread("myimage.jpg",0) #grayscale
    h,w = im.shape
    f = 700
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)
'''
def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0,0]

    im_h,im_w = img1.shape[:2]

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros((im_h,im_w), dtype=img1.dtype)
    cyl_h, cyl_w = cyl.shape[:2]
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0,cyl_w):
        for y_cyl in np.arange(0,cyl_h):
            theta = (x_cyl - x_c) / f
            h     = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K,X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
            cyl_mask[int(y_cyl),int(x_cyl)] = 255


    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png",bbox_inches='tight')

    return (cyl, cyl_mask)


if __name__ == "__main__":
    im = cv2.imread('../data/example-data/flower/1.jpg')
    h,w = im.shape[:2]
    f = 700
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)

    cv2.imshow("test", imcyl[0])
    cv2.waitKey()
    cv2.destroyAllWindows()
