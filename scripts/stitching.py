import numpy as np
import cv2
from features import FeatureExtractor
from matches import FeatureMatcher
from cylindrical import cylindricalWarpImage
from spherical import warpSpherical
from timeit import default_timer as timer

class Stitcher:
    def __init__(self, image_names=[], f=800, mode='Spherical'):
        self.__images = []
        self.__image_masks = []
        self.__features = []
        self.__image_pairs = []
        self.__start = None
        self.extractor = FeatureExtractor()
        self.matcher = FeatureMatcher()

        for image_name in image_names:
            img = cv2.imread(image_name)

            if mode=='Cylindrical':
                self.__transform_method = 'affine'
                #### convert rectangler to cylindrical
                h,w = img.shape[:2]
                start=timer()
                K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
                cylindrical_img, cylindrical_mask = cylindricalWarpImage(img, K)
                end=timer()
                print("convert time", end-start)
                print("converting", image_name, "to cylindrical ... ")
                # print("cylindrical mask dtype:", cylindrical_mask.dtype)
                # print("cylindrical mask shape:", cylindrical_mask.shape)

                self.__images.append(cylindrical_img)
                self.__image_masks.append(cylindrical_mask)

            if mode=='Spherical':
                self.__transform_method = 'affine'
                #### convert rectangler to spherical
                # f = 3000
                start=timer()
                spherical_img, spherical_mask = warpSpherical(img, f)
                end=timer()
                print("convert time", end-start)
                print("converting", image_name, "to spherical ... ")

                self.__images.append(spherical_img)
                self.__image_masks.append(spherical_mask)

            if mode=='Flat':
                self.__transform_method = 'homography'
                #### flat
                self.__images.append(img)
                self.__image_masks.append(np.ones(img.shape[:2], dtype=np.uint8)*255)

    def prepare_features(self):
        self.__features = []
        for i in range(len(self.__images)):
            # convert images to grayscale
            gray = cv2.cvtColor(self.__images[i], cv2.COLOR_RGB2GRAY)

            # extract features from images
            kp, des = self.extractor.getFeatures(gray)

            # store features
            feature = dict(kp=kp, des=des)
            self.__features.append(feature)

    def is_all_connected(self, all_nodes, connection, s, nodes_visited=set()):
        nodes_visited.add(s)
        for node in connection[s]:
            if node not in nodes_visited:
                self.is_all_connected(all_nodes, connection, node, nodes_visited)

        # print("nodes_visited in is_all_connected", nodes_visited)
        # print("all_nodes in is_all_connected", all_nodes)
        # print("edges in is_all_connected", connection)
        if nodes_visited==all_nodes:
            return True
        else:
            return False

    def bfs_find_pairs(self, connection, s, good_matrix, nodes_visited=set()):
        nodes_visited.add(s)
        for node in connection[s]:
            if node not in nodes_visited:
                pair_details = dict(src_index=s,
                                    dst_index=node,
                                    good_matched_points=good_matrix[s][node])
                self.__image_pairs.append(pair_details)
                self.bfs_find_pairs(connection, node, good_matrix, nodes_visited)

    def find_most_pair_two_images(self):
        #### prepare features
        if self.__features == []:
            self.prepare_features()

        #### initializa
        pair_matrix = np.zeros((len(self.__images), len(self.__images)))
        good_matrix = [[ None for _ in range(len(self.__images))]
                       for _ in range(len(self.__images))]

        #### find matches amount matrix
        print("matching features ...")
        for pre_i in range(len(self.__images)):
            for i in range(len(self.__images)):
                if pre_i==i:
                    continue

                print("matching features on image %d and image %d ..." % (pre_i, i))

                # matches feature
                # good = self.matcher.match(
                #     self.__features[pre_i]['des'], self.__features[i]['des']) # ->
                good = self.matcher.match(
                    self.__features[i]['des'], self.__features[pre_i]['des'])
                pair_matrix[pre_i, i] = len(good)
                good_matrix[pre_i][i] = good

        # print(pair_matrix)

        #### find the most pair image relationship
        image_number = len(pair_matrix)
        maximum_index = np.argmax(pair_matrix)
        row = maximum_index // image_number
        col = maximum_index % image_number
        # print("maximum_index =", maximum_index, "len(pair_matrix) =", image_number)
        # print("row =", row, "col =", col)
        # print("The maximum value is", pair_matrix[row][col])

        pair_matrix[row][col] = -1

        all_nodes = set([i for i in range(image_number)])
        nodes = set()
        nodes.add(row)
        nodes.add(col)

        edges = {}
        edges[row] = set()
        edges[row].add(col)
        edges[col] = set()
        edges[col].add(row)

        # print("all_nodes =", all_nodes)
        # print("nodes =", nodes)
        # print("edges =", edges)

        self.__start = row
        # print("Is a closed tree?", self.is_all_connected(all_nodes, edges, self.__start))
        while not self.is_all_connected(all_nodes, edges, self.__start, set()):
            maximum_index = np.argmax(pair_matrix)
            row = maximum_index // image_number
            col = maximum_index % image_number
            # print()
            # print("maximum_index =", maximum_index, "len(pair_matrix) =", image_number)
            # print("row =", row, "col =", col)
            # print("The maximum value is", pair_matrix[row][col])

            pair_matrix[row][col] = -1

            if row not in nodes:
                edges[row] = set()
            if col not in nodes:
                edges[col] = set()
            edges[row].add(col)
            edges[col].add(row)

            nodes.add(row)
            nodes.add(col)

            # print("all_nodes =", all_nodes)
            # print("nodes =", nodes)
            # print("edges =", edges)

        # print("Is a closed tree?", self.is_all_connected(all_nodes, edges, self.__start))

        #### construct self.__image_pairs
        self.bfs_find_pairs(edges, self.__start, good_matrix, set())


        # #### find the most pair image relationship (old)
        # pair_amount_sum = np.sum(pair_matrix, axis=1)
        # # print(pair_amount_sum)
        # ordered_index = np.argsort(pair_amount_sum)
        # # ordered_index = np.flip(ordered_index, axis=0)
        # # print(ordered_index)
        # img_index = ordered_index[0]
        # self.__start = img_index
        # for i in range(len(ordered_index)-1):
        #
        #     max_index = np.argmax(pair_matrix[img_index])
        #     # while max_index<i:# max_index in most_pair_image:
        #     #     pair_matrix[img_index][max_index] = 0
        #     #     max_index = np.argmax(pair_matrix[img_index])
        #     most_pair_image[img_index] = max_index
        #     pair_matrix[max_index][img_index] = 0 # avoid linked back
        #     pair_details = dict(src_index=img_index,
        #                         dst_index=max_index,
        #                         good_matched_points=good_matrix[img_index][max_index])
        #     self.__image_pairs.append(pair_details)
        #
        #     # print(pair_matrix)
        #     # print()
        #     img_index = max_index
        # return most_pair_image

    def find_Hs(self):
        # Initialize
        Hs = []
        # prepare pair details
        start = timer()

        if self.__image_pairs == []:
            self.find_most_pair_two_images()
        end = timer()
        print("Find Pair time:",end-start)
        # find Homography matrixes
        for i in range(len(self.__image_pairs)):
            pair = self.__image_pairs[i]
            index1 = pair['src_index']
            index2 = pair['dst_index']
            kp1 = self.__features[index1]['kp']
            kp2 = self.__features[index2]['kp']
            des1 = self.__features[index1]['des']
            des2 = self.__features[index2]['des']

            # get homography
            print("Finding transform for image %d and image %d" % (index1, index2))
            # H, _ = self.matcher.getTransform(kp1, kp2, des1, des2,
            #                                  good=pair['good_matched_points']) # ->
            H, _ = self.matcher.getTransform(kp2, kp1, des2, des1,
                                             good=pair['good_matched_points'],
                                             type=self.__transform_method)

            pair['H'] = H
            Hs.append(H)

        # print(self.__image_pairs)
        return Hs

    def stitch_all(self):
        if self.__start is None or self.__image_pairs==[] or 'H' not in self.__image_pairs[0].keys():
            self.find_Hs()

        shifted_H = np.eye(3)
        img1 = self.__images[self.__start]
        mask1 = self.__image_masks[self.__start]
        shifted_Hs = {}
        shifted_Hs[self.__start] = shifted_H
        print("Start stiching!")
        for i in range(len(self.__image_pairs)):
            pair = self.__image_pairs[i]

            index1 = pair['src_index']
            index2 = pair['dst_index']

            img2 = self.__images[index2]
            mask2 = self.__image_masks[index2]
            print("stiching image %d and image %d" % (index1,index2))


            # H = np.dot(shifted_H, last_H) # ->
            H = np.dot(pair['H'], np.linalg.inv(shifted_Hs[index1])) # <-
            # H = pair['H']


            # img1, shifted_H = self.mix_images(img1, img2, H, type=self.__transform_method) # ->
            img1, mask1, shifted_H = self.mix_images(img2, img1, mask2, mask1, H,
                                                     type=self.__transform_method)

            #### update cumulative translation/transform
            for k, v in shifted_Hs.items():
                shifted_Hs[k] = np.dot(H, shifted_H)
            if index2 not in shifted_Hs.keys():
                shifted_Hs[index2] = shifted_H
            shifted_Hs[index1] = np.dot(H, shifted_H)

        stiched_image = img1
        return stiched_image

    def old_stitch_all(self):
        new_img = None
        for image in self.__images:
            if new_img is None:
                new_img = image
            else:
                # convert images to grayscale
                new_img_gray = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                # extract features from images
                new_kp, new_des = self.extractor.getFeatures(new_img_gray)
                kp, des = self.extractor.getFeatures(img_gray)

                # matches feature and get homography
                H, _ = self.matcher.getTransform(kp, new_kp, des, new_des)
                # H, _ = self.matcher.getTransform(new_kp, kp, new_des, des)
                if H is None:
                    continue

                # stich images
                new_img, _, _ = stitcher.mix_images(image, new_img, H)
                # new_img, _ = sticher.mix_images(new_img, image, H)

        return new_img

    def mix_images(self, img1, img2, mask1, mask2, H, type="affine"):
        """
        This function will stick img2 to img1
        """
        h, w = img2.shape[:2]
        pts = np.float32([[0,0],
                          [0,h-1],
                          [w-1,h-1],
                          [w-1,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, H)
        # print("H:", H)
        # print("pts:", pts)
        # print("dst:", dst)

        h1, w1 = img1.shape[:2]

        maxs = np.max(dst, axis=0).flatten()
        mins = np.min(dst, axis=0).flatten()
        # print("Maxs:", maxs)
        # print("Mins", mins)

        abs_maxs = np.max([maxs, [w1, h1]], axis=0)
        abs_mins = np.min([mins, [0, 0]], axis=0)
        # print("Absolute minimum:", abs_mins)

        shift_amount = np.ceil([0, 0] - abs_mins).astype(np.int)
        # print("Shift amount:", shift_amount)

        shift_H = cv2.getPerspectiveTransform(
            np.float32([ [0,0],[0,10],[10,10],[10,0] ]).reshape(-1,1,2),
            np.float32([ [shift_amount[0],shift_amount[1]],
                         [shift_amount[0],shift_amount[1]+10],
                         [shift_amount[0]+10,shift_amount[1]+10],
                         [shift_amount[0]+10,shift_amount[1]] ]).reshape(-1,1,2)
        )
        # print("shift_H:", shift_H)

        new_image_size = tuple(np.ceil(abs_maxs - abs_mins).astype(np.int))
        # print("New image size:", new_image_size)

        #### wrap images
        this_shift_H = shift_H
        if type=="homography":
            new_img2 = cv2.warpPerspective(img2, np.dot(shift_H, H), new_image_size)
            new_mask2 = cv2.warpPerspective(mask2, np.dot(shift_H, H), new_image_size)
        if type=="affine":
            new_img2 = cv2.warpAffine(img2, np.dot(shift_H, H)[:2,:], new_image_size)
            new_mask2 = cv2.warpAffine(mask2, np.dot(shift_H, H)[:2,:], new_image_size)

        new_img1 = cv2.warpPerspective(img1, shift_H, new_image_size)
        new_mask1 = cv2.warpPerspective(mask1, shift_H, new_image_size)

        #### blend images
        cross_mask = cv2.bitwise_and(new_mask1, new_mask2)
        M = cv2.moments(cross_mask)

        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # print("cx:", cx, "cy", cy)

            new_img = np.maximum(new_img1, new_img2)
            new_img = cv2.seamlessClone(new_img1, new_img, cross_mask, (cx, cy), cv2.NORMAL_CLONE)
        except ZeroDivisionError:
            new_img = np.maximum(new_img1, new_img2)

        # # maximum pixel value blending
        # new_img = np.maximum(new_img1, new_img2)

        # # average blending
        # new_img = ((new_img1.astype(np.uint16) + new_img2.astype(np.uint16)) // 2).astype(np.uint8)

        # # no blending
        # new_img = new_img2.copy()
        # new_img[shift_amount[1]:h+shift_amount[1], shift_amount[0]:w+shift_amount[0]] = new_img1[shift_amount[1]:h+shift_amount[1], shift_amount[0]:w+shift_amount[0]]

        new_img_mask = cv2.bitwise_or(new_mask1, new_mask2)
        return new_img, new_img_mask, this_shift_H


if __name__ == "__main__":
    """ Here is a simple example of using this sticher """
    """
    # load images
    img1 = cv2.imread('../data/example-data/flower/1.jpg')
    img2 = cv2.imread('../data/example-data/flower/2.jpg')
    img3 = cv2.imread('../data/example-data/flower/3.jpg')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)

    # extract features from images
    extractor = FeatureExtractor()
    kp1, des1 = extractor.getFeatures(gray1)
    kp2, des2 = extractor.getFeatures(gray2)
    kp3, des3 = extractor.getFeatures(gray3)

    # matches feature and get homography
    matcher = FeatureMatcher()
    H, _ = matcher.getTransform(kp1, kp2, des1, des2)

    # stitch the first image and the second image
    stitcher = Stitcher()
    new_img = stitcher.mix_images(img1, img2,
                                  255*np.ones(img1.shape[:2], dtype=img1.dtype),
                                  255*np.ones(img2.shape[:2], dtype=img2.dtype),
                                  H)

    # stich the second image and the third image
    new_img_gray = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
    new_kp, new_des = extractor.getFeatures(new_img_gray)
    H, _ = matcher.getTransform(new_kp, kp3, new_des, des3)
    new_img = sticher.mix_images(new_img, img3,
                                 255*np.ones(new_img.shape[:2], dtype=new_img.dtype),
                                 255*np.ones(img3.shape[:2], dtype=img3.dtype),
                                 H)
    """

    stitcher = Stitcher([
        '../data/example-data/flower/1.jpg',
        '../data/example-data/flower/2.jpg',
        '../data/example-data/flower/3.jpg',
        '../data/example-data/flower/4.jpg',
        ],
        mode='Flat',
        )
    new_img = stitcher.stitch_all()

    cv2.imwrite("new_img.jpg", new_img)
