import cv2
import numpy
import sys
import numpy as np

import plotly.express as px
import matplotlib.pyplot as plt

IMAGE_PATH = r'C:\Users\eviatar\Desktop\eviatar\Study\YearD\semester b\VAN\VAN_ex\docs'
DATA_PATH = r'C:/Users/eviatar/Desktop/eviatar/Study/YearD/semester b/VAN/VAN_ex/dataset/sequences/00/'
FIRST_IMAGE = 000000
THRESH = 2

ORANGE = (255, 127, 14)
CYAN = (0, 255, 255)

numpy.set_printoptions(threshold=sys.maxsize)


# ex1 utils:
# -------------------------------------------------------------------------------------------------------------------
def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + 'image_0/' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + 'image_1/' + img_name, 0)
    return img1, img2


def detect_and_describe(img1, img2):
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    return kp1, des1, kp2, des2, img1, img2


def present_key_points(img1, kp1, img2, kp2):
    img_kp1 = cv2.drawKeypoints(img1, kp1, cv2.DRAW_MATCHES_FLAGS_DEFAULT, color=(120, 157, 187))
    img_kp2 = cv2.drawKeypoints(img2, kp2, cv2.DRAW_MATCHES_FLAGS_DEFAULT, color=(120, 157, 187))

    cv2.imwrite("img1_key_points.jpg", img_kp1)
    cv2.imwrite("img2_key_points.jpg", img_kp2)
    cv2.waitKey(0)


def match(kp1, des1, kp2, des2, img1, img2):
    matches = bf.match(des1, des2)
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, sorted_matches[:50], img2, flags=2)
    cv2.imwrite("img3_matches.jpg", img3)
    return img3, matches


def present_match(img3):
    cv2.imshow('SIFT', img3)
    cv2.waitKey(0)


# ex2:
# -------------------------------------------------------------------------------------------------------------------
def plot_deviation_from_stereo_pattern():
    matches_i_in_img1 = [m.queryIdx for m in matches]
    matches_i_in_img2 = [m.trainIdx for m in matches]
    deviations = []
    counter = 0
    for i, j in zip(matches_i_in_img1, matches_i_in_img2):
        abs_deviation = abs(key_points1[i].pt[1] - key_points2[j].pt[1])
        deviations.append(abs_deviation)
        if abs_deviation > 2:
            counter += 1

    fig = px.histogram(deviations, nbins=100)
    fig.update_xaxes(title_text='Deviation from rectified stereo pattern')
    fig.update_yaxes(title_text='Number of matches')

    # fig.show()
    print(f"Percentage of matches that deviate by more than 2 pixels: {counter * 100 / len(matches)}")


def reject_matches():
    matches_i_in_img1 = [m.queryIdx for m in matches]
    matches_i_in_img2 = [m.trainIdx for m in matches]
    image1_inliers = []
    image2_inliers = []
    image1_outlyers = []
    image2_outlyers = []

    for i, j in zip(matches_i_in_img1, matches_i_in_img2):
        if abs(key_points1[i].pt[1] - key_points2[j].pt[1]) < THRESH:
            image1_inliers.append(key_points1[i])
            image2_inliers.append(key_points2[j])
        else:
            image1_outlyers.append(key_points1[i])
            image2_outlyers.append(key_points2[j])
    print(len(image1_inliers))
    print(len(image2_inliers))

    return [image1_inliers, image2_inliers], [image1_outlyers, image2_outlyers]


def draw_rejected_matches(inliers, outlyers):
    rec_img_in = cv2.drawMatches(image1, inliers[0], image2, inliers[1], [], flags=cv2.DrawMatchesFlags_DEFAULT,
                                 outImg=None,
                                 singlePointColor=ORANGE)
    rec_img_out = cv2.drawMatches(image1, outlyers[0], image2, outlyers[1], [],
                                  flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG, outImg=rec_img_in,
                                  singlePointColor=CYAN)
    cv2.imwrite("rec_img_inout.jpg", rec_img_out)


if __name__ == '__main__':
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    image1, image2 = read_images(FIRST_IMAGE)
    key_points1, descriptor1, key_points2, descriptor2, image1, image2 = detect_and_describe(image1, image2)
    # present_key_points(image1, key_points1, image2, key_points2)
    image3, matches = match(key_points1, descriptor1, key_points2, descriptor2, image1, image2)
    # present_match(image3)
    # print_descriptors(descriptor1[1], descriptor2[1])

    # 2.1:
    plot_deviation_from_stereo_pattern()

    # 2.2:
    in_liers, out_liers = reject_matches()
    draw_rejected_matches(in_liers, out_liers)
