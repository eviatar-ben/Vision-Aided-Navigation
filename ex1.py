import cv2

DATA_PATH = r"C:\Users\eviatar\Desktop\eviatar\Study\YearD\semester b\VAN\VAN_ex\dataset\sequences\00"


def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + 'image_0/' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + 'image_1/' + img_name, 0)
    return img1, img2


def detect_and_describe(img1, img2):
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    return kp1, des1, kp2, des2


def present(kp1, des1, kp2, des2):
    pass


if __name__ == '__main__':
    orb = cv2.ORB_create()
    sift = cv2.SIFT_create()
    image1, image2 = read_images()
    detect_and_describe(image1, image2)
