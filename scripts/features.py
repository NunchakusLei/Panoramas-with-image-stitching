import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self, type='SIFT'):
        if type=='SIFT':
            self.extractor = cv2.xfeatures2d.SIFT_create()
        elif type=='SURF':
            self.extractor = cv2.xfeatures2d.SURF_create()
        elif type=='ORB':
            self.extractor = cv2.ORB_create()
        else:
            raise typeError("Error, unknown feature type!")

    def getFeatures(self, img):
        kp, des = self.extractor.detectAndCompute(img, None)
        return kp, des

if __name__ == "__main__":
    img = cv2.imread('../data/example-data/flower/1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # extractor = FeatureExtractor('SURF')
    extractor = FeatureExtractor('SIFT')
    kp, des = extractor.getFeatures(gray)
    labeled_img = cv2.drawKeypoints(
        img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("features", labeled_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
