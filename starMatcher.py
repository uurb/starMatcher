import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


def starMatcher(img1, img2, verbose=False):

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=10000, edgeThreshold=10, fastThreshold=10)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    ## match descriptors and sort them in the order of their distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    ## extract the matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    ## find homography matrix and do perspective transform
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    print(dst)

    if verbose:
        ## draw found region
        show_img = cv2.polylines(np.array(img2).copy(), [np.int32(dst)], True, (255, 255, 255), 1, cv2.LINE_AA)
        plt.figure()
        plt.imshow(show_img, cmap="gray")
        plt.show()

    return dst


if __name__ == '__main__':
    argument_count = len(sys.argv)
    if not(argument_count == 3):
        print("usage: python starMatcher.py imagePath1 imagePath2")
        exit()
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]

    img1 = cv2.imread(image1_path, -1)
    img2 = cv2.imread(image2_path, -1)
    starMatcher(img1, img2)

