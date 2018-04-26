import numpy as np
import cv2
from features import FeatureExtractor

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 0
LOWES_RATIO = 0.5

class FeatureMatcher:
    def __init__(self):
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, des1, des2):
        matches = self.matcher.knnMatch(des2, des1, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < LOWES_RATIO*n.distance:
                good.append(m)

        return good

    def getTransform(self, kp1, kp2, des1, des2, good=None, type='homography'):
        if good is None:
            good = self.match(des1, des2)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            if type=='homography':
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

            if type=='affine':
                M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
                M = np.append(M, [[0,0,1]], axis=0)

            # matchesMask = mask.ravel().tolist()
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            M, mask = None, None

        return M, mask


if __name__ == "__main__":
    # load images
    img1 = cv2.imread('../data/example-data/flower/1.jpg')
    img2 = cv2.imread('../data/example-data/flower/2.jpg')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # extract features from images
    extractor = FeatureExtractor()
    kp1, des1 = extractor.getFeatures(gray1)
    kp2, des2 = extractor.getFeatures(gray2)

    # matches feature and get homography
    matcher = FeatureMatcher()
    H, _ = matcher.getTransform(kp1, kp2, des1, des2)

    # display the results
    print("Homography matrix:")
    print(H)
    print("%d good matched points found." % len(matcher.match(des1, des2)))
