import cv2
import ex1
import ex2
import numpy as np

k, m1, m2 = ex2.get_camera_mat()


def get_mutual_kp(matches00, matches11, matches01):
    result1 = []
    result0 = []
    ml0 = matches00.keys() & matches01.keys()
    for i in ml0:
        if matches01[i] in matches11:
            result1.append(matches01[i])
            result0.append(i)
    return result1, result0


def get_p3d(des1, des2, mutual, matches):
    left0, left1 = [], []
    for i in mutual:
        left0.append(des1[i])
        left1.append(des2[matches[i]])
    # todo maybe triangulate each in oder to maintain order
    return ex2.triangulation([left0, left1], m1, m2)


def get_pl1(mutual_kp, kp):
    result = []
    for i in mutual_kp:
        result.append(kp[i])
    return np.array(result)


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


if __name__ == '__main__':
    l0, r0 = ex1.read_images(ex1.FIRST_IMAGE)
    l1, r1 = ex1.read_images(ex1.SECOND_IMAGE)
    # 3.1:
    matches00p, kpL0 = ex2.get_matches_stereo(l0, r0)
    matches11p, kpL1 = ex2.get_matches_stereo(l1, r1)
    img0_cloud, img1_cloud = ex2.get_cloud(ex2.FIRST_IMAGE), ex2.get_cloud(ex2.SECOND_IMAGE)
    # 3.2:
    key_points1, key_points2, matches01p = ex1.get_significance_matches(image1=l0, image2=l1)
    # 3.3:
    mutual_kpL1, mutual_kpL0 = get_mutual_kp(matches00p, matches11p, matches01p)
    p3d = get_p3d(key_points1, key_points2, mutual_kpL0, matches00p)
    pl1 = get_pl1(mutual_kpL1, kpL1)
    # todo: check for correlation between indices
    a = p3d[:4]
    b = cv2.KeyPoint_convert(pl1[:4])
    suc, r, t = cv2.solvePnP(a, b, cameraMatrix=k, distCoeffs=None, flags=cv2.SOLVEPNP_AP3P)
    if not suc:
        print("solvePnP failed")
    Rt = rodriguez_to_mat(r, t)
    print(Rt)
