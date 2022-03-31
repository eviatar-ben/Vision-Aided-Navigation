import random

import cv2
import numpy
import sys

IMAGE_PATH = r'C:\Users\eviatar\Desktop\eviatar\Study\YearD\semester b\VAN\VAN_ex\docs'
DATA_PATH = r'C:/Users/eviatar/Desktop/eviatar/Study/YearD/semester b/VAN/VAN_ex/dataset/sequences/00/'
FIRST_IMAGE = 000000
numpy.set_printoptions(threshold=sys.maxsize)


def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + 'image_0/' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + 'image_1/' + img_name, 0)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2


def detect_and_describe(img1, img2):
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    return kp1, des1, kp2, des2, img1, img2


def match(kp1, des1, kp2, des2, img1, img2, random_20_matches=True):
    if random_20_matches:
        random_matches = random.choices(bf.match(des1, des2), k=20)
        sorted_matches = sorted(random_matches, key=lambda x: x.distance)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, sorted_matches[:50], img2, flags=2)
        cv2.imwrite("img3_matches.jpg", img3)
    else:
        matches = bf.match(des1, des2)
        sorted_matches = sorted(matches, key=lambda x: x.distance)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, sorted_matches[:50], img2, flags=2)
        cv2.imwrite("img3_matches.jpg", img3)
    return img3


def present_key_points(img1, kp1, img2, kp2):
    img_kp1 = cv2.drawKeypoints(img1, kp1, cv2.DRAW_MATCHES_FLAGS_DEFAULT, color=(120, 157, 187))
    img_kp2 = cv2.drawKeypoints(img2, kp2, cv2.DRAW_MATCHES_FLAGS_DEFAULT, color=(120, 157, 187))

    cv2.imwrite("img1_key_points.jpg", img_kp1)
    cv2.imwrite("img2_key_points.jpg", img_kp2)
    cv2.waitKey(0)


def present_match(img3):
    # cv2.imwrite(IMAGE_PATH, img3)
    cv2.imshow('SIFT', img3)
    cv2.waitKey(0)


def print_descriptors(des1, des2):
    print(f"Image1's second feature descriptor is:\n {des1}")
    print(f"Image2's second feature descriptor is:\n {des2}")


def analyse_matches(dsc1, dsc2, ratio):
    two_nn = bf_ncc.knnMatch(dsc1[:500], dsc2[:500], k=2)
    passed = 0
    for f, s in two_nn:
        if f.distance / s.distance < ratio:
            passed += 1

    return passed


if __name__ == '__main__':
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf_ncc =cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    image1, image2 = read_images(FIRST_IMAGE)
    key_points1, descriptor1, key_points2, descriptor2, image1, image2 = detect_and_describe(image1, image2)
    present_key_points(image1, key_points1, image2, key_points2)
    analyse_matches(descriptor1, descriptor2, 0.3)
    image3 = match(key_points1, descriptor1, key_points2, descriptor2, image1, image2)
    present_match(image3)
    # print_descriptors(descriptor1[1], descriptor2[1])
