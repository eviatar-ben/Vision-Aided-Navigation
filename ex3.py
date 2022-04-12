import cv2
import ex1
import ex2
import ex3_tests
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


THRESH =2

k, km1, km2, m1, m2 = ex2.get_camera_mat()


def get_mutual_kp(matches00, matches11, matches01):
    result1 = []
    result0 = []
    ml0 = matches00.keys() & matches01.keys()
    for i in ml0:
        if matches01[i] in matches11.keys():
            result1.append(matches01[i])
            result0.append(i)
    return result0, result1


def get_p3d(kp1, kp2, mutual, matches):
    left0, left1 = [], []
    for i in mutual:
        left0.append(kp1[i])
        left1.append(kp2[matches[i]])
    # todo maybe triangulate each in oder to maintain order
    # return ex2.cv2_triangulation([left0, left1], m1, m2)
    return ex2.triangulation([left0, left1], km1, km2)


def get_pl1(mutual_kp, kp):
    result = []
    for i in mutual_kp:
        result.append(kp[i])
    return np.array(result)


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def extract_fours(mutual_kpL0, mutual_kpL1, kpL0, kpR0, kpL1, kpR1, matches00p, matches11p):
    l_0, r_0, l_1, r_1 = [], [], [], []
    for i, j in zip(mutual_kpL0, mutual_kpL1):
        l_0.append(kpL0[i].pt)
        r_0.append(kpR0[matches00p[i]].pt)

        l_1.append(kpL1[j].pt)
        r_1.append(kpR1[matches11p[j]].pt)

    return l_0, r_0, l_1, r_1


def plot_cmr_relative_position(ext_r0, ext_l1, ext_r1):
    l0cam = np.asarray([0, 0, 0])
    r0cam = -np.linalg.inv(ext_r0[:, :-1]) @ ext_r0[:, -1]
    # r0cam = -m2[:, -1]
    l1cam = -np.linalg.inv(ext_l1[:, :-1]) @ ext_l1[:, -1]
    r1cam = -np.linalg.inv(ext_r1[:, :-1]) @ ext_r1[:, -1]
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d', title="The relative position of the four cameras")

    ax.scatter(l0cam[0], l0cam[2])
    ax.scatter(r0cam[0], r0cam[2])
    ax.scatter(l1cam[0], l1cam[2])
    ax.scatter(r1cam[0], r1cam[2])
    # plt.show()

    # TODO: SHOT NEED TO BE FROM ABOVE
    # fig = px.scatter_3d(p, x=0, y=1, z=2, labels={
    #     '0': "X axis",
    #     '1': "Y axis",
    #     '2': "Z axis"},
    #                     title=f'Calculated 3D points yields by linear least squares triangulation ({algorithm})')
    # fig.show()


def calc_mat(T):
    ext_l0 = np.column_stack((np.identity(3), np.zeros(shape=(3, 1))))
    ext_l1, ext_r0 = T, m2
    r1r2 = T[:, :-1] @ (m2[:, :-1])
    r2t1_t2 = (m2[:, :-1]) @ (T[:, -1]) + m2[:, -1]
    ext_r1 = np.column_stack((r1r2, r2t1_t2))
    return ext_l0, ext_r0, ext_l1, ext_r1


def projection(ext_l0, ext_r0, ext_l1, ext_r1, p3d):
    # pl0, pr0, pl1, pr1
    # pl0 = k@p3d
    # pl0 /= pl0[-1, :]
    #
    # # pr0 = k@ ext_r0 @ p3d
    # pr0 = km2@p3d
    # pr0 /= pr0[-1, :]
    #
    # # pl1 = k@T @p3d
    # pl1 = k@ext_l1 @p3d
    # pl1 /= pl1[-1, :]
    #
    # pr1 = k@ext_r1@p3d
    # pr1 /= pr1[-1, :]

    pl0 = p3d @ ext_l0.T @ k.T
    pr0 = p3d @ ext_r0.T @ k.T
    pl1 = p3d @ ext_l1.T @ k.T
    pr1 = p3d @ ext_r1.T @ k.T

    pl0 = pl0[:, :2].T / pl0[:, -1]
    pl0 = pl0.T
    pr0 = pr0[:, :2].T / pr0[:, -1]
    pr0 = pr0.T
    pl1 = pl1[:, :2].T / pl1[:, -1]
    pl1 = pl1.T
    pr1 = pr1[:, :2].T / pr1[:, -1]
    pr1 = pr1.T

    return pl0, pr0, pl1, pr1


def transform3dp(p3d):
    return np.column_stack([p3d, np.ones(len(p3d))])


def get_supporters(pl0, pr0, pl1, pr1, lm0, rm0, lm1, rm1):
    supoortl0 = np.power(lm0 - pl0, 2).sum(axis=0) <= THRESH ** 2
    supoortr0 = np.power(rm0 - pr0, 2).sum(axis=0) <= THRESH ** 2
    supoortl1 = np.power(lm1 - pl1, 2).sum(axis=0) <= THRESH ** 2
    supoortr1 = np.power(rm1 - pr1, 2).sum(axis=0) <= THRESH ** 2
    return

def main():
    l0, r0 = ex1.read_images(ex1.FIRST_IMAGE)
    l1, r1 = ex1.read_images(ex1.SECOND_IMAGE)
    # 3.1:
    matches00p, kpL0, kpR0 = ex2.get_matches_stereo(l0, r0)
    matches11p, kpL1, kpR1 = ex2.get_matches_stereo(l1, r1)
    img0_cloud, img1_cloud = ex2.get_cloud(ex2.FIRST_IMAGE), ex2.get_cloud(ex2.SECOND_IMAGE)
    # 3.2:
    matches01p, _, _ = ex1.get_significance_matches(img1=l0, img2=l1)  # todo: change the parameter for efficiency
    # 3.3:
    mutual_kpL0, mutual_kpL1 = get_mutual_kp(matches00p, matches11p, matches01p)
    # tests:

    lm0, rm0, lm1, rm1 = extract_fours(mutual_kpL0, mutual_kpL1, kpL0, kpR0, kpL1, kpR1, matches00p, matches11p)
    # ex3_tests.draw_tracking(l0, r0, l1, r1, lm0[:4], rm0[:4], lm1[:4], rm1[:4])

    p3d = get_p3d(kpL0, kpR0, mutual_kpL0, matches00p)

    pl1 = get_pl1(mutual_kpL1, kpL1)
    # todo: check for correlation between indices between 3d points
    a = p3d[:4]
    b = cv2.KeyPoint_convert(pl1[:4])
    suc, r, t = cv2.solvePnP(a, b, cameraMatrix=k, distCoeffs=None, flags=cv2.SOLVEPNP_AP3P)
    if not suc:
        print("solvePnP failed")
    Rt = rodriguez_to_mat(r, t)
    # todo: finish the plot

    ext_l0, ext_r0, ext_l1, ext_r1 = calc_mat(Rt)
    plot_cmr_relative_position(ext_r0, ext_l1, ext_r1)
    # 2.4:
    p3d = transform3dp(p3d)
    pl0, pr0, pl1, pr1 = projection(ext_l0, ext_r0, ext_l1, ext_r1, p3d)

    get_supporters(pl0, pr0, pl1, pr1, lm0, rm0, lm1, rm1)


if __name__ == '__main__':
    main()
