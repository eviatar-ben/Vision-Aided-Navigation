import cv2

IMAGE_PATH = r'C:\Users\eviatar\Desktop\eviatar\Study\YearD\semester b\VAN\VAN_ex\docs'
DATA_PATH = r'C:/Users/eviatar/Desktop/eviatar/Study/YearD/semester b/VAN/VAN_ex/dataset/sequences/00/'
FIRST_IMAGE = 000000


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
    return kp1, des1, kp2, des2, img1, img2


def match(kp1, des1, kp2, des2, img1, img2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, sorted_matches[:50], img2, flags=2)
    return img3


def present(img3):
    # cv2.imwrite(IMAGE_PATH, img3)
    cv2.imshow('ORB', img3)
    cv2.waitKey(0)


def print_first_descriptors(f_des1, f_des2):
    print(f"Image1's first feature descriptor is {des1}")
    print(f"Image2's first feature descriptor is {des2}")


if __name__ == '__main__':
    orb = cv2.ORB_create()
    image1, image2 = read_images(FIRST_IMAGE)
    kp1, des1, kp2, des2, img1, img2 = detect_and_describe(image1, image2)
    image3 = match(kp1, des1, kp2, des2, img1, img2)
    present(image3)
    print_first_descriptors(des1[0], des2[0])
