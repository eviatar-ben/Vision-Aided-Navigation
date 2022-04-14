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
GROUND_TRUTH_PATH = r"../VAN_ex/dataset/poses/00.txt'"

k, km1, km2, m1, m2 = ex2.get_camera_mat()


def get_mutual_kp_ind(matches00, matches11, matches01):
    mutual_kp_ind_l0 = []
    mutual_kp_ind_l1 = []
    ml0 = matches00.keys() & matches01.keys()
    for i in ml0:
        if matches01[i] in matches11.keys():
            mutual_kp_ind_l0.append(i)
            mutual_kp_ind_l1.append(matches01[i])
    return mutual_kp_ind_l0, mutual_kp_ind_l1


def get_p3d(kp1, kp2, mutual, matches):
    left0, left1 = [], []
    for i in mutual:
        left0.append(kp1[i])
        left1.append(kp2[matches[i]])
    # todo maybe triangulate each in oder to maintain order
    # return ex2.triangulation([left0, left1], m1, m2)
    return ex2.cv2_triangulation([left0, left1], km1, km2)


def get_pl1(mutual_kp, kp):
    result = []
    for i in mutual_kp:
        result.append(kp[i])
    return np.array(result)


def get_pl0(mutual_matches_ind_l1, kp_l0, matches01p):
    result = []
    rev_matches01p = {y: x for x, y in matches01p.items()}
    for i in mutual_matches_ind_l1:
        result.append(kp_l0[matches01p[i]])
    return np.array(result)


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def extract_fours(mutual_kpL0, mutual_kpL1, kpL0, kpR0, kpL1, kpR1, matches00p, matches11p):
    point_l1, point_r1 = [], []
    # p_l0, p_r0, p_l1, p_r1 = [], [], [], []

    # todo check the counter and the conversion
    for j in mutual_kpL1:
        point_l1.append(kpL1[j].pt)

        point_r1.append(kpR1[matches11p[j]].pt)

    return point_l1, point_r1


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
    r2r1 = m2[:, :-1] @ (T[:, :-1])
    r2t1_t2 = (m2[:, :-1]) @ (T[:, -1]) + m2[:, -1]
    ext_r1 = np.column_stack((r2r1, r2t1_t2))
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


def get_supporters(projected_l1, projected_r1, matched_l1, matched_r1):
    support_l1 = np.power(matched_l1 - projected_l1, 2).sum(axis=1) <= THRESH ** 2
    support_r1 = np.power(matched_r1 - projected_r1, 2).sum(axis=1) <= THRESH ** 2

    # support_l1_ind = np.asarray(support_l1).nonzero()
    # support_r1_ind = np.asarray(support_r1).nonzero()

    # supporters =
    # intersect supporters:
    # support_l1_ind = {convert_pos__idx_l1[i] for i in support_l1_ind[0]}
    # support_r1_ind = {convert_pos__idx_r1[i] for i in support_r1_ind[0]}
    # support_r1_ind = {matches11p[i] for i in support_r1_ind}

    return np.logical_and(support_r1, support_l1).nonzero()


def plot_supporters(l0, l1, supporters, pl1, pl0):
    support_l0 = []
    support_l1 = []
    pl1 = pl1.tolist()
    for i in supporters[0]:
        support_l0.append(pl0[i].pt)
        support_l1.append(pl1[i].pt)

    # print(support_l0)
    # print(support_l1)

    support_l0_x = [i[0] for i in support_l0]
    support_l0_y = [i[1] for i in support_l0]

    support_l1_x = [i[0] for i in support_l1]
    support_l1_y = [i[1] for i in support_l1]
    # todo : complete the plot

    fig = plt.figure(figsize=(10, 7))
    rows, cols = 2, 1
    fig.suptitle(f'')

    # Left1 camera
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(l1, cmap='gray')
    ax1.set_title("Left0 camera")
    ax1.scatter(support_l0_x, support_l0_y, s=1, color="cyan")
    # plt.scatter(left1_matches_coor[supporters_idx][:, 0],
    #             left1_matches_coor[supporters_idx][:, 1], s=3, color="orange")

    # Left0 camera
    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(l0, cmap='gray')
    ax2.set_title("Left1 camera")
    ax2.scatter(support_l1_x, support_l1_y, s=1, color="cyan")
    # plt.scatter(left0_matches_coor[supporters_idx][:, 0],
    #             left0_matches_coor[supporters_idx][:, 1], s=3, color="orange")

    fig.savefig("supporters.png")
    plt.close(fig)


def get_maximal_group(p3d, pl1, point_l1, point_r1):
    maximum = 0
    maximum_supporters_idx = None
    # for _ in range(int(len(p3d) / RANDOM_FACTOR)):
    for _ in range(int(len(p3d) * 2)):
        i = np.random.randint(len(p3d), size=4)
        object_points, image_points = p3d[i], cv2.KeyPoint_convert(pl1[i])
        suc, r, t = cv2.solvePnP(object_points, image_points, cameraMatrix=k, distCoeffs=None, flags=cv2.SOLVEPNP_AP3P)
        try:
            Rt = rodriguez_to_mat(r, t)
        except:
            continue
        ext_l0, ext_r0, ext_l1, ext_r1 = calc_mat(Rt)
        projected_l1, projected_r1 = projection(ext_l1, ext_r1, transform3dp(p3d))

        supporters_idx = get_supporters(projected_l1, projected_r1, point_l1, point_r1)
        if len(supporters_idx[0]) > maximum:
            maximum = len(supporters_idx[0])
            maximum_supporters_idx = supporters_idx
    return maximum_supporters_idx


def refine_transformation(supporters_idx, p3d, pl1):
    if not supporters_idx:
        return None
    object_points, image_points = p3d[supporters_idx], cv2.KeyPoint_convert(pl1[supporters_idx])
    try:
        suc, r, t = cv2.solvePnP(object_points, image_points, cameraMatrix=k, distCoeffs=None,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    except:
        return None
    if not suc:
        return None
    Rt = rodriguez_to_mat(r, t)
    return Rt


def transform_cloud(p3d, Rt_transpose):
    return p3d @ Rt_transpose


def plot_clouds(p3d, transform_p3d):
    fig = px.scatter_3d(p3d, x=0, y=1, z=2, labels={
        '0': "X axis",
        '1': "Y axis",
        '2': "Z axis"},
                        title=f'')
    fig.show()
    fig = px.scatter_3d(transform_p3d, x=0, y=1, z=2, labels={
        '0': "X axis",
        '1': "Y axis",
        '2': "Z axis"},
                        title=f'')
    fig.show()
    pass


def one_shot(i):
    l0, r0 = ex1.read_images(ex1.FIRST_IMAGE + i)
    l1, r1 = ex1.read_images(ex1.FIRST_IMAGE + (i + 1))
    kernel_size = 10
    l0 = cv2.blur(l0, (kernel_size, kernel_size))
    r0 = cv2.blur(r0, (kernel_size, kernel_size))
    l1 = cv2.blur(l1, (kernel_size, kernel_size))
    r1 = cv2.blur(r1, (kernel_size, kernel_size))

    match0, matches00p, kp_l0, kp_ro = ex2.get_matches_stereo(l0, r0)
    match11, matches11p, kp_l1, kp_r1 = ex2.get_matches_stereo(l1, r1)
    matches01p, _, _ = ex1.get_significance_matches(img1=l0, img2=l1)
    # match01, matches01p, _, _ = ex2.get_brute_force_matches(img1=l0, img2=l1)
    mutual_matches_ind_l0, mutual_matches_ind_l1 = get_mutual_kp_ind(matches00p, matches11p, matches01p)

    p3d = get_p3d(kp_l0, kp_ro, mutual_matches_ind_l0, matches00p)
    pl1 = get_pl1(mutual_matches_ind_l1, kp_l1)

    point_l1, point_r1 = extract_fours(mutual_matches_ind_l0, mutual_matches_ind_l1, kp_l0, kp_ro, kp_l1, kp_r1,
                                       matches00p, matches11p)

    supporters_idx = get_maximal_group(p3d, pl1, np.asarray(point_l1), np.asarray(point_r1))
    Rt = refine_transformation(supporters_idx, p3d, pl1)
    return Rt


def play(stop):
    def compute_rts():
        for i in range(0, stop):
            print(i)
            rts_path.append(one_shot(i))

    def compute_relative_transformation():
        def get_composition(trans1, trans2):
            # R2R1 * v + R2t1 + t2.
            r2r1 = trans2[:, :-1] @ (trans1[:, :-1])
            r2t1_t2 = (trans2[:, :-1]) @ (trans1[:, -1]) + trans2[:, -1]
            ext_r1 = np.column_stack((r2r1, r2t1_t2))
            return ext_r1

        last = m1
        for trans in rts_path:
            last = get_composition(last, trans)
            relative_transformation_path.append(last)

    def compute_positions():
        for trans in relative_transformation_path:
            positions.append(-np.linalg.inv(trans[:, :-1]) @ trans[:, -1])

    rts_path = []
    relative_transformation_path = []
    positions = []

    compute_rts()
    compute_relative_transformation()
    compute_positions()
    return np.array(positions)


def plot_positions(positions):
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory for {len(positions)} frames.")
    ax.scatter(positions[:, 0], positions[:, 2], s=1, c='red')

    fig.savefig(f"Trajectory_from_cmd.png")
    plt.close(fig)


def get_ground_truth_transformations(left_cam_trans_path=GROUND_TRUTH_PATH):
    trans_ground_truth = []
    with open(left_cam_trans_path) as f:
        lines = f.readlines()
    for i in range(3450):
        left_mat = np.array(lines[i].split(" "))[:-1].astype(float).reshape((3, 4))
        trans_ground_truth.append(left_mat)
    return trans_ground_truth


# def main():
#     l0, r0 = ex1.read_images(ex1.FIRST_IMAGE)
#     l1, r1 = ex1.read_images(ex1.SECOND_IMAGE)
#     # 3.1:
#     match0, matches00p, kp_l0, kp_ro = ex2.get_matches_stereo(l0, r0)
#     match11, matches11p, kp_l1, kp_r1 = ex2.get_matches_stereo(l1, r1)
#     img0_cloud, img1_cloud = ex2.get_cloud(ex2.FIRST_IMAGE), ex2.get_cloud(ex2.SECOND_IMAGE)
#     # 3.2:
#     matches01p, _, _ = ex1.get_significance_matches(img1=l0, img2=l1)  # todo: change the parameter for efficiency
#
#     # match01, matches01p, _, _ = ex2.get_brute_force_matches(img1=l0,
#     #                                                         img2=l1)  # todo: change the parameter for efficiency
#     # 3.3:
#     mutual_matches_ind_l0, mutual_matches_ind_l1 = get_mutual_kp_ind(matches00p, matches11p, matches01p)
#
#     # tests:
#     # lm0, rm0, lm1, rm1 = extract_fours(mutual_matches_ind_l0, mutual_matches_ind_l1, kpL0, kpR0, kpL1, kpR1,
#     # matches00p, matches11p)
#     # ex3_tests.draw_tracking(l0, r0, l1, r1, lm0[70:74], rm0[70:74], lm1[70:74], rm1[70:74])
#     # ex3_tests.test_mutual(mutual_matches_ind_l0, mutual_matches_ind_l1, matches00p, matches11p, matches01p)
#
#     p3d = get_p3d(kp_l0, kp_ro, mutual_matches_ind_l0, matches00p)
#
#     pl1 = get_pl1(mutual_matches_ind_l1, kp_l1)
#     # todo: test that
#     pl0 = get_pl0(mutual_matches_ind_l0, kp_l0, matches01p)
#
#     object_points, image_points = p3d[0:4], cv2.KeyPoint_convert(pl1[0:4])
#     suc, r, t = cv2.solvePnP(object_points, image_points, cameraMatrix=k, distCoeffs=None, flags=cv2.SOLVEPNP_AP3P)
#     Rt = rodriguez_to_mat(r, t)
#
#     # todo: finish the plot
#     ext_l0, ext_r0, ext_l1, ext_r1 = calc_mat(Rt)
#     plot_cmr_relative_position(ext_r0, ext_l1, ext_r1)
#
#     # 2.4:
#     point_l1, point_r1 = extract_fours(mutual_matches_ind_l0, mutual_matches_ind_l1, kp_l0, kp_ro, kp_l1, kp_r1,
#                                        matches00p, matches11p)
#
#     projected_l1, projected_r1 = projection(ext_l1, ext_r1, transform3dp(p3d))
#
#     supporters = get_supporters(projected_l1, projected_r1, np.asarray(point_l1), np.asarray(point_r1))
#     # todo: finish the plot maybe th problem is only in pl0 values conversion from pl1
#     plot_supporters(l0, l1, supporters, pl1, pl0)
#
#     # 2.5:
#     supporters_idx = get_maximal_group(p3d, pl1, np.asarray(point_l1), np.asarray(point_r1))
#     Rt = refine_transformation(supporters_idx, p3d, pl1)
#
#     transform_p3d = transform_cloud(transform3dp(p3d), Rt.T)
#     # todo need to merge both clouds together
#     # plot_clouds(p3d, transform_p3d)
#     # 2.5:
#     # play(500)


if __name__ == '__main__':
    # main()
    positions = play(1000)
    plot_positions(positions)
