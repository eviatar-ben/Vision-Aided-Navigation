import random
import cv2
import numpy
import sys
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = r"C:\Users\eviatar\Desktop\eviatar\Study\YearD\semester b\VAN\VAN_ex\dataset\sequences\00"
import matplotlib.pyplot as plt

IMAGE_PATH = r'C:\Users\eviatar\Desktop\eviatar\Study\YearD\semester b\VAN\VAN_ex\docs'
DATA_PATH = r'C:/Users/eviatar/Desktop/eviatar/Study/YearD/semester b/VAN/VAN_ex/dataset/sequences/00/'
FIRST_IMAGE = 000000
RATIO = 0.25  # equalibrium point
numpy.set_printoptions(threshold=sys.maxsize)


def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + 'image_0/' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + 'image_1/' + img_name, 0)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2


def detect_and_describe(img1, img2):
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    return kp1, des1, kp2, des2
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    return kp1, des1, kp2, des2, img1, img2


def present_key_points(img1, kp1, img2, kp2):
    img_kp1 = cv2.drawKeypoints(img1, kp1, cv2.DRAW_MATCHES_FLAGS_DEFAULT, color=(120, 157, 187))
    img_kp2 = cv2.drawKeypoints(img2, kp2, cv2.DRAW_MATCHES_FLAGS_DEFAULT, color=(120, 157, 187))

    cv2.imwrite("img1_key_points.jpg", img_kp1)
    cv2.imwrite("img2_key_points.jpg", img_kp2)
    cv2.waitKey(0)


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


def present_match(img3):
    cv2.imshow('SIFT', img3)
    cv2.waitKey(0)


def print_descriptors(des1, des2):
    print(f"Image1's second feature descriptor is:\n {des1}")
    print(f"Image2's second feature descriptor is:\n {des2}")


def analyse_matches(query_descriptors, train_descriptors, factor, plot=False):
    """
    lowe's-ratio-test-work :
    1. If the "good" match can't be distinguished from noise (second match),
    then the "good" match should be rejected because it does not bring anything interesting, information-wise.
    2. doing distance1 - distance2 would be less robust,
    would require frequent tweaking and would make methodological comparisons more complicated.
    It's all about the ratio.
    :param plot:
    :param query_descriptors:
    :param train_descriptors:
    :param factor: multiply s.distance by a constant that has to be between 0 and 1,
     thus decreasing the value of distance2
    :return: number of
    """
    # knnMatch function will return the matches from best to worst, so the s.distance match will have a smaller distance
    two_nn = bf_ncc.knnMatch(query_descriptors, train_descriptors, k=2)
    filtered_noise = 0
    filtered = []
    unfiltered = []
    for f, s in two_nn:
        # multiply f.distance by a constant that has to be between 0 and 1,
        # thus decreasing the value of s.distance
        if f.distance / s.distance < factor:
            filtered_noise += 1
            filtered.append([f])
        else:
            unfiltered.append([f])
    if plot:
        print(
            f"With ratio value equals to {factor}: {len(unfiltered)} matches were discarded, and {len(filtered)} not.")
    return filtered_noise, filtered, unfiltered


def plot_ratios(dsc1, dsc2):
    x = np.arange(0, 1, 0.05)
    y = []
    z = []
    for f in x:
        filtered = analyse_matches(dsc1, dsc2, f)[0]
        y.append(filtered)
        z.append(len(dsc1) - filtered)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=x, y=y, name="Distinguished matches"),
        secondary_y=False)

    fig.add_trace(go.Scatter(x=x, y=z, name="Undistinguished matches"), secondary_y=True, )

    fig.update_layout(title_text="Distinguished matches in function of ratio")

    fig.update_xaxes(title_text="Ratio values")

    fig.update_yaxes(title_text="<b>Distinguished noises</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Undistinguished noises</b>", secondary_y=True)
    fig.show()

    # px.scatter(x=x, y=y,
    #            title="Distinguished matches in function of ratio: ",
    #            labels=dict(x="Ratio", y="Distinguished matches")).show()
    # px.scatter(x=x, y=z,
    #            title="Undistinguished matches in function of ratio: ",
    #            labels=dict(x="Ratio", y="Undistinguished matches")).show()


def present_significance_test(kp1, img1, kp2, img2, passed):
    random_matches = random.choices(passed, k=20)
    sorted_matches = sorted(random_matches, key=lambda x: x[0].distance)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, sorted_matches, None, flags=2)
    cv2.imwrite("img3_passed_matches.jpg", img3)
    return img3



def present(kp1, des1, kp2, des2):
    pass


if __name__ == '__main__':
    orb = cv2.ORB_create()
    sift = cv2.SIFT_create()
    image1, image2 = read_images()
    detect_and_describe(image1, image2)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    bf_ncc = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    image1, image2 = read_images(FIRST_IMAGE)
    key_points1, descriptor1, key_points2, descriptor2, image1, image2 = detect_and_describe(image1, image2)
    present_key_points(image1, key_points1, image2, key_points2)
    # plot_ratios(descriptor1, descriptor2)
    image3 = match(key_points1, descriptor1, key_points2, descriptor2, image1, image2)
    # present_match(image3)
    # print_descriptors(descriptor1[1], descriptor2[1])
    filtered_amount, passed_kp, failed_kp = analyse_matches(descriptor1, descriptor2, RATIO, plot=True)
    present_significance_test(key_points1, image1, key_points2, image2, passed_kp)