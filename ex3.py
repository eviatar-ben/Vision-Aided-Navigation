import cv2
import numpy as np
import ex1
import ex2
import exs_plots
import time
import pickle

THRESH = 2
RANDOM_FACTOR = 16
FRAMES_NUM = 3450

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
    # return ex2.triangulation([left0, left1], m1, m2)
    return ex2.cv2_triangulation([left0, left1], km1, km2)


def get_mutual_kp_l1(mutual_kp, kp):
    result = []
    for i in mutual_kp:
        result.append(kp[i])
    return np.array(result)


def get_mutual_kp_l0(mutual_matches_ind_l1, kp_l0, matches01p):
    mutual_matches_kp_l0 = []
    rev_matches10p = {y: x for x, y in matches01p.items()}
    for i in mutual_matches_ind_l1:
        mutual_matches_kp_l0.append(kp_l0[rev_matches10p[i]])
    return np.array(mutual_matches_kp_l0)


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def get_pl1_and_pr1(mutual_kpL0, mutual_kpL1, kpL0, kpR0, kpL1, kpR1, matches00p, matches11p):
    point_l1, point_r1 = [], []
    # p_l0, p_r0, p_l1, p_r1 = [], [], [], []

    # todo check the counter and the conversion
    for j in mutual_kpL1:
        point_l1.append(kpL1[j].pt)

        point_r1.append(kpR1[matches11p[j]].pt)

    return point_l1, point_r1


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
    # supporters are determined by norma 2.:
    support_l1 = np.power(matched_l1 - projected_l1, 2).sum(axis=1) <= THRESH ** 2
    support_r1 = np.power(matched_r1 - projected_r1, 2).sum(axis=1) <= THRESH ** 2

    return np.logical_and(support_r1, support_l1).nonzero()


def get_maximal_group(p3d, pl1, point_l1, point_r1):
    maximum = 0
    maximum_supporters_idx = None
    for _ in range(int(len(p3d) / RANDOM_FACTOR)):
        # for _ in range(int(len(p3d) * 16)):
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


def online_ransac(p3d, pl1, point_l1, point_r1):
    max_supporters_number, maximum_supporters_idx = -1, None
    inliers_num, outliers_num = 0, 0
    first_loop_iter = 0
    first_loop_iter_est = lambda prob, outliers_perc: np.log(1 - prob) / np.log(
        1 - np.power(1 - outliers_perc, 4))
    outliers_perc, prob = 0.99, 0.99

    while outliers_perc != 0 and first_loop_iter < first_loop_iter_est(prob, outliers_perc):
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
        num_supp = len(supporters_idx[0])

        if len(supporters_idx[0]) > max_supporters_number:
            max_supporters_number = len(supporters_idx[0])
            maximum_supporters_idx = supporters_idx

        first_loop_iter += 1

        outliers_num += len(p3d) - num_supp
        inliers_num += num_supp
        outliers_perc = min(outliers_num / (inliers_num + outliers_num), 0.99)
    return maximum_supporters_idx


def refine_transformation(supporters_idx, p3d, pl1):
    Rt = None
    try:
        object_points, image_points = p3d[supporters_idx], cv2.KeyPoint_convert(pl1[supporters_idx])
        suc, r, t = cv2.solvePnP(object_points, image_points, cameraMatrix=k, distCoeffs=None,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
        Rt = rodriguez_to_mat(r, t)
    except:
        print(f"refine_transformation failed with {supporters_idx} as args for cv2.solvePnP()")
    return Rt


def transform_cloud(p3d, Rt_transpose):
    return p3d @ Rt_transpose


def one_shot(i):
    l0, r0 = ex1.read_images(ex1.FIRST_IMAGE + i)
    l1, r1 = ex1.read_images(ex1.FIRST_IMAGE + (i + 1))

    # in the sake of efficiency blurring the images
    kernel_size = 10
    l0 = cv2.blur(l0, (kernel_size, kernel_size))
    r0 = cv2.blur(r0, (kernel_size, kernel_size))
    l1 = cv2.blur(l1, (kernel_size, kernel_size))
    r1 = cv2.blur(r1, (kernel_size, kernel_size))

    match0, matches00p, kp_l0, kp_r0 = ex2.get_matches_stereo(l0, r0)
    match11, matches11p, kp_l1, kp_r1 = ex2.get_matches_stereo(l1, r1)
    matches01p, _, _ = ex1.get_significance_matches(img1=l0, img2=l1)
    # match01, matches01p, _, _ = ex2.get_brute_force_matches(img1=l0, img2=l1)
    mutual_matches_ind_l0, mutual_matches_ind_l1 = get_mutual_kp_ind(matches00p, matches11p, matches01p)

    p3d = get_p3d(kp_l0, kp_r0, mutual_matches_ind_l0, matches00p)
    mutual_kp_l1 = get_mutual_kp_l1(mutual_matches_ind_l1, kp_l1)

    point_l1, point_r1 = get_pl1_and_pr1(mutual_matches_ind_l0, mutual_matches_ind_l1, kp_l0, kp_r0, kp_l1, kp_r1,
                                         matches00p, matches11p)

    # supporters_idx = get_maximal_group(p3d, mutual_kp_l1, np.asarray(point_l1), np.asarray(point_r1))
    supporters_idx = online_ransac(p3d, mutual_kp_l1, np.asarray(point_l1), np.asarray(point_r1))
    # supporters_idx is indices relevant to mutual_kp_l1
    # todo supporters_idx is indices relevant to mutual_kp_l1 are not relevant to kp_l0??
    Rt = refine_transformation(supporters_idx, p3d, mutual_kp_l1)

    first_frame_kp, second_frame_kp, supporters_matches01p = \
        get_l0_kp_in_frame(supporters_idx[0], mutual_matches_ind_l1, mutual_matches_ind_l0, matches01p,
                           matches00p, matches11p, kp_l0, kp_r0, kp_l1, kp_r1)
    inliers_per = len(supporters_idx[0]) / len(mutual_matches_ind_l0)  # supporters / consensus match
    return Rt, first_frame_kp, second_frame_kp, supporters_matches01p, inliers_per


def get_l0_kp_in_frame(supporters_idx, mutual_matches_ind_l1, mutual_matches_ind_l0, matches01p, matches00p, matches11p,
                       kp_l0, kp_r0, kp_l1, kp_r1, sanity_check=False):
    matches10p = {val: key for key, val in matches01p.items()}  # reverse 01 to 10
    # todo: check this out- bugs prone
    # 1. supporters_idx are relevant to l1
    # 2. [matches00p[i] for i in mutual_matches_ind_l0] need to be changed
    mutual_supporters_idx_matches_ind_l0 = [mutual_matches_ind_l0[i] for i in supporters_idx]
    mutual_supporters_idx_matches_ind_r0 = [matches00p[i] for i in mutual_supporters_idx_matches_ind_l0]
    mutual_supporters_idx_matches_ind_l1 = [matches01p[i] for i in mutual_supporters_idx_matches_ind_l0]
    mutual_supporters_idx_matches_ind_r1 = [matches11p[i] for i in mutual_supporters_idx_matches_ind_l1]

    if sanity_check:
        for i, j in zip(mutual_matches_ind_l0, mutual_matches_ind_l1):
            assert j == matches01p[i]

    mutual_supporters_kp_matches_ind_l0 = [kp_l0[i] for i in mutual_supporters_idx_matches_ind_l0]
    mutual_supporters_kp_matches_ind_r0 = [kp_r0[i] for i in mutual_supporters_idx_matches_ind_r0]
    mutual_supporters_kp_matches_ind_l1 = [kp_l1[i] for i in mutual_supporters_idx_matches_ind_l1]
    mutual_supporters_kp_matches_ind_r1 = [kp_r1[i] for i in mutual_supporters_idx_matches_ind_r1]

    first_frame_kp = (mutual_supporters_kp_matches_ind_l0, mutual_supporters_kp_matches_ind_r0)
    second_frame_kp = (mutual_supporters_kp_matches_ind_l1, mutual_supporters_kp_matches_ind_r1)

    supporters_matches01p = {key: matches01p[key] for key in mutual_supporters_idx_matches_ind_l0}

    return first_frame_kp, second_frame_kp, supporters_matches01p


def play(stop, pickling=True):

    import tqdm
    def compute_rts():
        for i in tqdm.tqdm(range(0, stop - 1)):
            Rt, first_frame_kp, second_frame_kp, supporters_matches01p, inliers_per = one_shot(i)
            rts_path.append(Rt)
            tracks_data.append([first_frame_kp, second_frame_kp, supporters_matches01p, inliers_per, Rt])

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
    tracks_data = []

    compute_rts()
    compute_relative_transformation()
    compute_positions()
    if pickling:
        tracks_data_with_points = []
        for i, data in enumerate(tracks_data):
            first_frame_kp, second_frame_kp, supporters_matches01p, inliers_per, Rt = \
                data[0], data[1], data[2], data[3], data[4]

            first_frame_p = cv2.KeyPoint_convert(np.asarray(first_frame_kp[0])), \
                            cv2.KeyPoint_convert(np.asarray(first_frame_kp[1]))
            second_frame_p = cv2.KeyPoint_convert(np.asarray(second_frame_kp[0])), cv2.KeyPoint_convert(
                np.asarray(second_frame_kp[1]))

            tracks_data_with_points.append([first_frame_p, second_frame_p, supporters_matches01p, inliers_per, Rt])

        # pickle_out = open(r"ex4_pickles\tracks_data.pickle", "wb")
        pickle_out = open(r"ex4_pickles/tracks_data.pickle", "wb")
        pickle.dump(tracks_data_with_points, pickle_out)
        pickle_out.close()

        pickle_out_transformations = open(r"ex4_pickles\global_transformations.pickle", "wb")
        pickle.dump(positions, pickle_out_transformations)
        pickle_out_transformations.close()
    return np.array(positions), tracks_data


def main():
    l0, r0 = ex1.read_images(ex1.FIRST_IMAGE)
    l1, r1 = ex1.read_images(ex1.SECOND_IMAGE)

    # 3.1:

    img0_cloud, img1_cloud = ex2.get_cloud(ex2.FIRST_IMAGE), ex2.get_cloud(ex2.SECOND_IMAGE)
    # exs_plots.plot_first_2_clouds(img0_cloud.T, img1_cloud.T)

    # 3.2:
    match0, matches00p, kp_l0, kp_ro = ex2.get_matches_stereo(l0, r0)
    match11, matches11p, kp_l1, kp_r1 = ex2.get_matches_stereo(l1, r1)
    matches01p, _, _ = ex1.get_significance_matches(img1=l0, img2=l1)  # todo: change the parameter for efficiency
    mutual_matches_ind_l0, mutual_matches_ind_l1 = get_mutual_kp_ind(matches00p, matches11p, matches01p)
    exs_plots.present_match_in_l0(kp_l0, mutual_matches_ind_l0, l0)

    # tests:
    # lm0, rm0, lm1, rm1 = extract_fours(mutual_matches_ind_l0, mutual_matches_ind_l1, kpL0, kpR0, kpL1, kpR1,
    # matches00p, matches11p)
    # ex3_tests.draw_tracking(l0, r0, l1, r1, lm0[70:74], rm0[70:74], lm1[70:74], rm1[70:74])
    # ex3_tests.test_mutual(mutual_matches_ind_l0, mutual_matches_ind_l1, matches00p, matches11p, matches01p)

    # 3.3:
    p3d = get_p3d(kp_l0, kp_ro, mutual_matches_ind_l0, matches00p)
    kpl1 = get_mutual_kp_l1(mutual_matches_ind_l1, kp_l1)

    object_points, image_points = p3d[0:4], cv2.KeyPoint_convert(kpl1[0:4])
    suc, r, t = cv2.solvePnP(object_points, image_points, cameraMatrix=k, distCoeffs=None, flags=cv2.SOLVEPNP_AP3P)
    Rt = rodriguez_to_mat(r, t)
    print(Rt)

    ext_l0, ext_r0, ext_l1, ext_r1 = calc_mat(Rt)
    exs_plots.plot_cmr_relative_position(ext_r0, ext_l1, ext_r1)

    # 3.4:
    point_l1, point_r1 = get_pl1_and_pr1(mutual_matches_ind_l0, mutual_matches_ind_l1, kp_l0, kp_ro, kp_l1, kp_r1,
                                         matches00p, matches11p)

    projected_l1, projected_r1 = projection(ext_l1, ext_r1, transform3dp(p3d))
    supporters = get_supporters(projected_l1, projected_r1, np.asarray(point_l1), np.asarray(point_r1))

    kpl0 = get_mutual_kp_l0(mutual_matches_ind_l1, kp_l0, matches01p)
    exs_plots.plot_supporters(l0, l1, supporters, kpl1, kpl0)

    # 3.5:
    # supporters_idx = get_maximal_group(p3d, kpl1, np.asarray(point_l1), np.asarray(point_r1))
    supporters_idx = online_ransac(p3d, kpl1, np.asarray(point_l1), np.asarray(point_r1))

    Rt = refine_transformation(supporters_idx, p3d, kpl1)

    transform_p3d = transform_cloud(transform3dp(p3d), Rt.T)
    # todo need to merge both clouds together
    exs_plots.plot_clouds(p3d, transform_p3d)
    exs_plots._plot_in_and_out_liers(l0, l1, supporters_idx, kpl1, kpl0)
    # 3.6:
    # positions = play(3450)
    # exs_plots.draw_left_cam_3d_trajectory(positions)


# ------------------------------------------------------ex4-----------------------------------------------------------
def get_frame_and_kps_l0_l1(supporters_idx, kp_l0, kp_r0, kp_l1, kp_r1, matches00p, matches01p, matches11p, i):
    pass


if __name__ == '__main__':
    # main()
    start_time = time.time()
    positions, _ = play(FRAMES_NUM)

    # exs_plots.plot_both_trajectories(positions)
    # exs_plots.draw_left_cam_3d_trajectory(positions)

    elapsed_time = time.time() - start_time
    print(elapsed_time)

# todo check the flann.knnmatch
