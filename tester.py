from starMatcher import starMatcher
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import math
import sys


def rotate_around_point(point, radians, origin):
    # Rotate a point around a given point.
    x, y = point
    ox, oy = origin

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)
    return qx, qy


def test(NumberOfSamples):

    image = cv2.imread('StarMap.png', -1)  # trainImage
    image_h, image_w = image.shape[:2]
    center_point = (image_w // 2, image_h // 2)

    # crate random patch values
    patch_x = np.random.randint(image_w//4, image_w//4 * 3, NumberOfSamples)
    patch_y = np.random.randint(image_h//4, image_h//4 * 3, NumberOfSamples)
    patch_w = np.random.randint(image_w//12, image_w//8, NumberOfSamples)
    patch_h = np.random.randint(image_h//12, image_h//8, NumberOfSamples)
    angle_list = np.random.randint(0, 360, NumberOfSamples)
    template_patch_values = list(zip(patch_x, patch_y, patch_w, patch_h, angle_list))

    positives = 0
    negatives = 0

    for index in range(NumberOfSamples):
        x, y, w, h, angle = template_patch_values[index]
        print("x, y, x+w, y+h, angle ", x, y, x+w, y+h, angle)

        rotated_image = imutils.rotate(image, angle)
        template_patch = rotated_image[y:y+h, x:x+w]

        result = starMatcher(template_patch, image, False)
        found_pt = result[0][0]
        rotated_topleft_point = rotate_around_point(found_pt, angle * (math.pi / 180), center_point)
        if abs(rotated_topleft_point[0] - x) < 20 and abs(rotated_topleft_point[1] - y) < 20:
            positives += 1
        else:
            negatives += 1
            print("rotated point: ", rotated_topleft_point)
        print("\n")

    print("##########################")
    print("out of", NumberOfSamples, "samples: \n")
    print("positives ", positives)
    print("negatives ", negatives)


if __name__ == '__main__':
    argument_count = len(sys.argv)
    if argument_count != 2:
        print("usage: python tester.py NumberOfSamples")
        exit()
    test(int(sys.argv[1]))

