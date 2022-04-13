import cv2
import ex1
import ex2
import ex3_tests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import plotly.express as px
import pandas as pd

THRESH = 2
RANDOM_FACTOR = 16
k, km1, km2, m1, m2 = ex2.get_camera_mat()


def get_mutual_kp_ind(matches00, matches11, matches01):
    mutual_kp_ind_l0 = []
    mutual_kp_ind_r0 = []
    ml0 = matches00.keys() & matches01.keys()
    for i in ml0:
        if matches01[i] in matches11.keys():
            mutual_kp_ind_l0.append(i)
            mutual_kp_ind_r0.append(matches01[i])
    return mutual_kp_ind_l0, mutual_kp_ind_r0


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
    point_l1, point_r1 = [], []
    # p_l0, p_r0, p_l1, p_r1 = [], [], [], []

    convert_pos_idx_l1 = {}
    convert_pos_idx_r1 = {}
    # todo check the counter and the conversion
    counter = 0
    for i, j in zip(mutual_kpL0, mutual_kpL1):
        # p_and_i_l0.append(kpL0[i].pt)
        # p_and_i_r0.append(kpR0[matches00p[i]].pt, matches00p[i]))

        point_l1.append(kpL1[j].pt)
        convert_pos_idx_l1[counter] = j

        point_r1.append(kpR1[matches11p[j]].pt)
        convert_pos_idx_r1[counter] = matches11p[j]

        counter += 1

    return convert_pos_idx_l1, convert_pos_idx_r1, point_l1, point_r1


def plot_cmr_relative_position(ext_r0, ext_l1, ext_r1):
    l0cam = np.asarray([0, 0, 0])
    r0cam = -np.linalg.inv(ext_r0[:, :-1]) @ ext_r0[:, -1]
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


def projection(ext_l1, ext_r1, p3d):
    pl1 = p3d @ ext_l1.T @ k.T
    pr1 = p3d @ ext_r1.T @ k.T

    pl1 = pl1[:, :2].T / pl1[:, -1]
    pl1 = pl1.T
    pr1 = pr1[:, :2].T / pr1[:, -1]
    pr1 = pr1.T

    return pl1, pr1


def transform3dp(p3d):
    return np.column_stack([p3d, np.ones(len(p3d))])


def get_supporters(projected_l1, projected_r1, matched_l1, matched_r1, convert_pos__idx_l1, convert_pos__idx_r1,
                   matches11p):
    intersect_supp_ind_l1 = []
    support_l1 = np.power(matched_l1 - projected_l1, 2).sum(axis=1) <= THRESH ** 2
    support_r1 = np.power(matched_r1 - projected_r1, 2).sum(axis=1) <= THRESH ** 2

    support_l1_ind = np.asarray(support_l1).nonzero()
    support_r1_ind = np.asarray(support_r1).nonzero()

    support_r1_ind = [convert_pos__idx_r1[i] for i in support_r1_ind[0]]

    # intersect supporters:
    rev_matches11p = {y: x for x, y in matches11p.items()}
    for i in support_l1_ind[0]:
        if rev_matches11p[convert_pos__idx_l1[i]] in support_r1_ind:
            pass

    return support_l1, support_r1


def plot_supporters(l0, l1, support_l1, support_r1, kp_l1, kp_r1, matches11p):
    support_l1_ind = support_l1.nonzero()
    support_r1_ind = support_r1.nonzero()

    support_l1 = []
    support_r1 = []

    for i, j in zip(support_l1_ind[0], support_r1_ind[0]):
        support_l1.append(kp_l1[i].pt)
        support_r1.append(kp_r1[j].pt)

    # todo : complete the plot

    fig = plt.figure(figsize=(10, 7))
    rows, cols = 2, 1
    fig.suptitle(f'supporters')
    plt.imshow(l1, cmap='gray')
    plt.title("Left1 camera")
    plt.scatter(support_l1[::2], support_l1[1::2], s=10, color="cyan")
    # plt.scatter(left1_matches_coor[supporters_idx][:, 0],
    #             left1_matches_coor[supporters_idx][:, 1], s=3, color="orange")

    fig.savefig("supporters.png")
    plt.close(fig)


# def get_maximal_group(p3d, pl1, lm1, rm1, matches11p,
#                       mutual_matches_ind_l0, mutual_matches_ind_l1):
#     maximum = 0
#     four = None
#     for _ in range(int(len(p3d) / RANDOM_FACTOR)):
#         i = np.random.randint(len(p3d), size=4)
#         object_points, image_points = p3d[i], cv2.KeyPoint_convert(pl1[i])
#         suc, r, t = cv2.solvePnP(object_points, image_points, cameraMatrix=k, distCoeffs=None, flags=cv2.SOLVEPNP_AP3P)
#         try:
#             Rt = rodriguez_to_mat(r, t)
#         except:
#             continue
#         ext_l0, ext_r0, ext_l1, ext_r1 = calc_mat(Rt)
#         projected_l1, projected_r1 = projection(ext_l1, ext_r1, transform3dp(p3d))
#         support_l1, support_r1 = get_supporters(projected_l1, projected_r1, np.asarray(lm1), np.asarray(rm1),
#                                                 matches11p,
#                                                 mutual_matches_ind_l0, mutual_matches_ind_l1)
#
#         sup_num = np.count_nonzero(support_l1) + np.count_nonzero(support_r1)
#         if sup_num > maximum:
#             maximum = sup_num
#             four = i
#     return maximum, four


def main():
    l0, r0 = ex1.read_images(ex1.FIRST_IMAGE)
    l1, r1 = ex1.read_images(ex1.SECOND_IMAGE)
    # 3.1:
    matches00p, kpL0, kpR0 = ex2.get_matches_stereo(l0, r0)
    matches11p, kpL1, kpR1 = ex2.get_matches_stereo(l1, r1)
    # img0_cloud, img1_cloud = ex2.get_cloud(ex2.FIRST_IMAGE), ex2.get_cloud(ex2.SECOND_IMAGE)
    # 3.2:
    matches01p, _, _ = ex1.get_significance_matches(img1=l0, img2=l1)  # todo: change the parameter for efficiency
    # 3.3:
    mutual_matches_ind_l0, mutual_matches_ind_l1 = get_mutual_kp_ind(matches00p, matches11p, matches01p)

    # tests:
    # lm0, rm0, lm1, rm1 = extract_fours(mutual_matches_ind_l0, mutual_matches_ind_l1, kpL0, kpR0, kpL1, kpR1, matches00p,
    #                                    matches11p)
    # ex3_tests.draw_tracking(l0, r0, l1, r1, lm0[70:74], rm0[70:74], lm1[70:74], rm1[70:74])

    p3d = get_p3d(kpL0, kpR0, mutual_matches_ind_l0, matches00p)

    pl1 = get_pl1(mutual_matches_ind_l1, kpL1)

    object_points, image_points = p3d[10:14], cv2.KeyPoint_convert(pl1[10:14])
    suc, r, t = cv2.solvePnP(object_points, image_points, cameraMatrix=k, distCoeffs=None, flags=cv2.SOLVEPNP_AP3P)
    Rt = rodriguez_to_mat(r, t)

    # todo: finish the plot
    ext_l0, ext_r0, ext_l1, ext_r1 = calc_mat(Rt)
    plot_cmr_relative_position(ext_r0, ext_l1, ext_r1)

    # 2.4:
    # todo: efficiency discard lm0, rm0
    convert_pos_idx_l1, convert_pos_idx_r1, point_l1, point_r1 = extract_fours(
        mutual_matches_ind_l0, mutual_matches_ind_l1, kpL0, kpR0, kpL1, kpR1, matches00p, matches11p)

    projected_l1, projected_r1 = projection(ext_l1, ext_r1, transform3dp(p3d))
    support_l1, support_r1 = get_supporters(projected_l1, projected_r1, np.asarray(point_l1),
                                            np.asarray(point_r1), convert_pos_idx_l1, convert_pos_idx_r1, matches11p)
    plot_supporters(l0, l1, np.asarray(support_l1), np.asarray(support_r1), kpL1, kpR1, matches11p)

    # 2.5:
    # maximum, four = get_maximal_group(p3d, pl1, lm1, rm1, matches11p,
    #                                   mutual_matches_ind_l0, mutual_matches_ind_l1)


if __name__ == '__main__':
    main()

# No need to project on the first pair: always will emits support
# def projection(ext_l0, ext_r0, ext_l1, ext_r1, p3d):
#     pl0 = p3d @ ext_l0.T @ k.T
#     pr0 = p3d @ ext_r0.T @ k.T
#     pl1 = p3d @ ext_l1.T @ k.T
#     pr1 = p3d @ ext_r1.T @ k.T
#
#     pl0 = pl0[:, :2].T / pl0[:, -1]
#     pl0 = pl0.T
#     pr0 = pr0[:, :2].T / pr0[:, -1]
#     pr0 = pr0.T
#     pl1 = pl1[:, :2].T / pl1[:, -1]
#     pl1 = pl1.T
#     pr1 = pr1[:, :2].T / pr1[:, -1]
#     pr1 = pr1.T
#
#     return pl0, pr0, pl1, pr1

# No need to project on the first pair: always will emits support
# def get_supporters(pl0, pr0, pl1, pr1, lm0, rm0, lm1, rm1):
#     supoortl0 = np.power(lm0 - pl0, 2).sum(axis=1) <= THRESH ** 2
#     supoortr0 = np.power(rm0 - pr0, 2).sum(axis=1) <= THRESH ** 2
#     supoortl1 = np.power(lm1 - pl1, 2).sum(axis=1) <= THRESH ** 2
#     supoortr1 = np.power(rm1 - pr1, 2).sum(axis=1) <= THRESH ** 2
#     return supoortl0, supoortr0, supoortl1, supoortr1
