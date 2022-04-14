import cv2
import numpy
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

IMAGE_PATH = r'C:\Users\eviatar\Desktop\eviatar\Study\YearD\semester b\VAN\VAN_ex\docs'
DATA_PATH = r'C:/Users/eviatar/Desktop/eviatar/Study/YearD/semester b/VAN/VAN_ex/dataset/sequences/00/'
FIRST_IMAGE = 000000
SECOND_IMAGE = 0o00001
THRESH = 2
IMG_NUM = 3

ORANGE = (255, 127, 14)
CYAN = (0, 255, 255)

numpy.set_printoptions(threshold=sys.maxsize)

sift = cv2.SIFT_create()


# ex1 utils:
# -------------------------------------------------------------------------------------------------------------------
def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + 'image_0/' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + 'image_1/' + img_name, 0)
    return img1, img2


def detect_and_describe(img1, img2, sift):
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
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
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
def plot_deviation_from_stereo_pattern(matches, key_points1, key_points2):
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


def reject_matches(matches, key_points1, key_points2):
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

    return [image1_inliers, image2_inliers], [image1_outlyers, image2_outlyers]


def draw_rejected_matches(image1, image2, inliers, outlyers):
    rec_img_in = cv2.drawMatches(image1, inliers[0], image2, inliers[1], [], flags=cv2.DrawMatchesFlags_DEFAULT,
                                 outImg=None,
                                 singlePointColor=ORANGE)
    rec_img_out = cv2.drawMatches(image1, outlyers[0], image2, outlyers[1], [],
                                  flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG, outImg=rec_img_in,
                                  singlePointColor=CYAN)
    cv2.imwrite("rec_img_inout.jpg", rec_img_out)


def read_cameras():
    with open(DATA_PATH + 'calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


def triangulation(in_liers, m1c, m2c):
    def compute_homogenates_matrix(p, q):
        p, q = p.pt, q.pt
        return np.array([p[0] * m1c[2] - m1c[0],
                         p[1] * m1c[2] - m1c[1],
                         q[0] * m2c[2] - m2c[0],
                         q[1] * m2c[2] - m2c[1]])

    xs = []
    for i, j in zip(in_liers[0], in_liers[1]):
        homogenates_matrix = compute_homogenates_matrix(i, j)
        _, _, vh = np.linalg.svd(homogenates_matrix)
        last_col = vh[-1]
        xs.append(last_col[:3] / last_col[3])
    return np.array(xs)


def cv2_triangulation(in_liers, m1c, m2c):
    def edit_to_pt():
        for i, j in zip(in_liers[0], in_liers[1]):
            points1.append(i.pt)
            points2.append(j.pt)

    points1 = []
    points2 = []
    edit_to_pt()

    ps = cv2.triangulatePoints(m1c, m2c, np.array(points1).T, np.array(points2).T).T
    return np.squeeze(cv2.convertPointsFromHomogeneous(ps))


def present_world_3d_points(points, cv2=False):
    if cv2:
        algorithm = 'Lecture algorithm'
    else:
        algorithm = 'cv2 algorithm'
    p = pd.DataFrame(points)
    fig = px.scatter_3d(p, x=0, y=1, z=2, labels={
        '0': "X axis",
        '1': "Y axis",
        '2': "Z axis"},
                        title=f'Calculated 3D points yields by linear least squares triangulation ({algorithm})')
    fig.show()


def get_camera_mat():
    k, m1, m2 = read_cameras()
    km1 = k @ m1
    km2 = k @ m2
    return k, km1, km2, m1, m2


# ----------------------------------------------------------------------------------------------------------------------
# ex3 utils:
def get_cloud(image):
    _, m1c, m2c, _, _ = get_camera_mat()

    image1, image2 = read_images(image)
    key_points1, descriptor1, key_points2, descriptor2, image1, image2 = detect_and_describe(image1, image2, sift)
    # present_key_points(image1, key_points1, image2, key_points2)
    image3, matches = match(key_points1, descriptor1, key_points2, descriptor2, image1, image2)
    # 2.2:
    in_liers, out_liers = reject_matches(matches, key_points1, key_points2)
    draw_rejected_matches(image1, image2, in_liers, out_liers)

    # 2.3A:
    world_3d_points = triangulation(in_liers, m1c, m2c)
    # cv2_world_3d_points = cv2_triangulation(in_liers, m1c, m2c)
    # present_world_3d_points(world_3d_points)
    # present_world_3d_points(cv2_world_3d_points)
    return world_3d_points


def rectify(matches, key_points1, key_points2):
    idx_kp1 = {}
    # todo check of query is frame1
    matches_i_in_img1 = [m.queryIdx for m in matches]
    matches_i_in_img2 = [m.trainIdx for m in matches]
    for i, j in zip(matches_i_in_img1, matches_i_in_img2):
        if abs(key_points1[i].pt[1] - key_points2[j].pt[1]) < THRESH:
            idx_kp1[i] = j
    return idx_kp1


def get_matches_stereo(image1, image2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    key_points1, descriptor1, key_points2, descriptor2, image1, image2 = detect_and_describe(image1, image2, sift)
    matches = bf.match(descriptor1, descriptor2)
    idx_kp1 = rectify(matches, key_points1, key_points2)
    return matches, idx_kp1, key_points1, key_points2


def get_brute_force_matches(img1, img2):
    def rectify2(matches):
        idx_kp1 = {}
        # todo check of query is frame1
        matches_i_in_img1 = [m.queryIdx for m in matches]
        matches_i_in_img2 = [m.trainIdx for m in matches]
        for i, j in zip(matches_i_in_img1, matches_i_in_img2):
            idx_kp1[i] = j
        return idx_kp1
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    key_points1, descriptor1, key_points2, descriptor2, image1, image2 = detect_and_describe(img1, img2, sift)
    matches = bf.match(descriptor1, descriptor2)
    idx_kp1 = rectify2(matches)
    return matches, idx_kp1, key_points1, key_points2


if __name__ == '__main__':
    # In retrospect the following rows commented in order to change the API for further exercise
    # sift = cv2.SIFT_create()
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # _, m1c, m2c = get_camera_mat()
    #
    # image1, image2 = read_images(FIRST_IMAGE)
    # key_points1, descriptor1, key_points2, descriptor2, image1, image2 = detect_and_describe(image1, image2)
    # # present_key_points(image1, key_points1, image2, key_points2)
    # image3, matches = match(key_points1, descriptor1, key_points2, descriptor2, image1, image2)
    # # present_match(image3)
    # # print_descriptors(descriptor1[1], descriptor2[1])
    #
    # # 2.1:
    # plot_deviation_from_stereo_pattern(matches, key_points1, key_points2)
    #
    # # 2.2:
    # in_liers, out_liers = reject_matches()
    # draw_rejected_matches(in_liers, out_liers)
    #
    # # 2.3A:
    # world_3d_points = triangulation()
    # cv2_world_3d_points = cv2_triangulation()
    # present_world_3d_points(world_3d_points)
    # present_world_3d_points(cv2_world_3d_points, True)
    # 2.3B:
    # run line 181 : image1, image2 = read_images(SECOND_IMAGE)
    # with argument SECOND_IMAGE instead of FIRST_IMAGE
    get_cloud(FIRST_IMAGE)
    get_cloud(SECOND_IMAGE)
