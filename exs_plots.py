import matplotlib.pyplot as plt
import cv2
import numpy as np

# from ex4 import FRAMES_NUM
import utilities

FRAMES_NUM = 3450


# -----------------------------------------------------3----------------------------------------------------------------
# 3.1
def plot_first_2_clouds(img0_cloud, img1_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', title="Two first images' clouds")
    ax.scatter(*img0_cloud, color='b')
    ax.scatter(*img1_cloud, color='g')
    fig.show()


# 3.2
def present_match_in_l0(kp, mutual_matches_ind_l0, l0):
    img_kp1 = cv2.drawKeypoints(l0, [kp[i] for i in mutual_matches_ind_l0], cv2.DRAW_MATCHES_FLAGS_DEFAULT,
                                color=(120, 157, 187))
    cv2.imwrite("plots/ex3/mutual_key_points.jpg", img_kp1)


# 3.3
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


# 3.4
def plot_supporters(l0, l1, supporters, pl1, pl0):
    def get_x_y_support_points(pl0, pl1, supporters):
        support_l0 = []
        support_l1 = []
        for i in supporters[0]:
            support_l0.append(pl0[i].pt)
            support_l1.append(pl1[i].pt)
        # print(support_l0)
        # print(support_l1)
        support_l0_x = [i[0] for i in support_l0]
        support_l0_y = [i[1] for i in support_l0]
        support_l1_x = [i[0] for i in support_l1]
        support_l1_y = [i[1] for i in support_l1]
        return support_l0_x, support_l0_y, support_l1_x, support_l1_y

    def get_x_y_non_support_points(pl0, pl1, supporters):
        nonsupport_l0 = []
        non_support_l1 = []
        unsupporters = {i for i in range(len(pl0))} - set(supporters[0].tolist())
        for i in unsupporters:
            nonsupport_l0.append(pl0[i].pt)
            non_support_l1.append(pl1[i].pt)
        unsupport_l0_x = [i[0] for i in nonsupport_l0]
        unsupport_l0_y = [i[1] for i in nonsupport_l0]
        unsupport_l1_x = [i[0] for i in non_support_l1]
        unsupport_l1_y = [i[1] for i in non_support_l1]
        return unsupport_l0_x, unsupport_l0_y, unsupport_l1_x, unsupport_l1_y

    non_support_l0_x, non_support_l0_y, non_support_l1_x, non_support_l1_y = get_x_y_non_support_points(pl0, pl1,
                                                                                                        supporters)

    support_l0_x, support_l0_y, support_l1_x, support_l1_y = get_x_y_support_points(pl0, pl1, supporters)
    # todo : complete the plot

    fig = plt.figure(figsize=(10, 7))
    rows, cols = 2, 1
    fig.suptitle(f'Supporters (using 2 pixels threshold) and non supporters')

    # Left1 camera
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(l0, cmap='gray')
    ax1.set_title("Left0 camera")
    ax1.scatter(support_l0_x, support_l0_y, s=1, color="cyan")
    ax1.scatter(non_support_l0_x, non_support_l0_y, s=1, color="orange")

    # plt.scatter(left1_matches_coor[supporters_idx][:, 0],
    #             left1_matches_coor[supporters_idx][:, 1], s=3, color="orange")

    # Left0 camera
    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(l1, cmap='gray')
    ax2.set_title("Left1 camera")
    ax2.scatter(support_l1_x, support_l1_y, s=1, color="cyan")
    ax2.scatter(non_support_l1_x, non_support_l1_y, s=1, color="orange")

    # plt.scatter(left0_matches_coor[supporters_idx][:, 0],
    #             left0_matches_coor[supporters_idx][:, 1], s=3, color="orange")

    fig.savefig(r"plots\ex3\supporters.png")
    plt.close(fig)


# 3.5:


def plot_clouds(p3d, transform_p3d):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = plt.axes(projection='3d')
    ax.set_title(f"3d Point clouds")
    ax.scatter3D(p3d[:, 0], p3d[:, 1], p3d[:, 2], c='red')
    ax.scatter3D(transform_p3d[:, 0], transform_p3d[:, 1], transform_p3d[:, 2], c='cyan')
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-100, 100)

    fig.savefig(r"plots\ex3\3d_clouds.png")
    plt.close(fig)


def _plot_in_and_out_liers(l0, l1, supporters_idx, pl1, pl0):
    def get_x_y_support_points(pl0, pl1, supporters):
        support_l0 = []
        support_l1 = []
        for i in supporters[0]:
            support_l0.append(pl0[i].pt)
            support_l1.append(pl1[i].pt)
        # print(support_l0)
        # print(support_l1)
        support_l0_x = [i[0] for i in support_l0]
        support_l0_y = [i[1] for i in support_l0]
        support_l1_x = [i[0] for i in support_l1]
        support_l1_y = [i[1] for i in support_l1]
        return support_l0_x, support_l0_y, support_l1_x, support_l1_y

    def get_x_y_non_support_points(pl0, pl1, supporters):
        nonsupport_l0 = []
        non_support_l1 = []
        unsupporters = {i for i in range(len(pl0))} - set(supporters[0].tolist())
        for i in unsupporters:
            nonsupport_l0.append(pl0[i].pt)
            non_support_l1.append(pl1[i].pt)
        unsupport_l0_x = [i[0] for i in nonsupport_l0]
        unsupport_l0_y = [i[1] for i in nonsupport_l0]
        unsupport_l1_x = [i[0] for i in non_support_l1]
        unsupport_l1_y = [i[1] for i in non_support_l1]
        return unsupport_l0_x, unsupport_l0_y, unsupport_l1_x, unsupport_l1_y

    non_support_l0_x, non_support_l0_y, non_support_l1_x, non_support_l1_y = get_x_y_non_support_points(pl0, pl1,
                                                                                                        supporters_idx)

    support_l0_x, support_l0_y, support_l1_x, support_l1_y = get_x_y_support_points(pl0, pl1, supporters_idx)
    # todo : complete the plot

    fig = plt.figure(figsize=(10, 7))
    rows, cols = 2, 1
    fig.suptitle(f'Inliers and Outliers in Images L0 and L1')

    # Left1 camera
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(l0, cmap='gray')
    ax1.set_title("Left0 camera")

    ax1.scatter(support_l0_x, support_l0_y, s=1, color="cyan")
    ax1.scatter(non_support_l0_x, non_support_l0_y, s=1, color="orange")

    # plt.scatter(left1_matches_coor[supporters_idx][:, 0],
    #             left1_matches_coor[supporters_idx][:, 1], s=3, color="orange")

    # Left0 camera
    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(l1, cmap='gray')
    ax2.set_title("Left1 camera")
    ax2.scatter(support_l1_x, support_l1_y, s=1, color="cyan")
    ax2.scatter(non_support_l1_x, non_support_l1_y, s=1, color="orange")

    # plt.scatter(left0_matches_coor[supporters_idx][:, 0],
    #             left0_matches_coor[supporters_idx][:, 1], s=3, color="orange")

    fig.savefig(r"plots\ex3\in_and_out_liers.png")
    plt.close(fig)


# 3.6:
def plot_trajectory_2d(positions):
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2D trajectory for {len(positions)} frames.")
    ax.scatter(positions[:, 0], positions[:, 2], s=1, c='red')

    fig.savefig(r"plots\ex3\Trajectory 2D.png")
    plt.close(fig)


def draw_left_cam_3d_trajectory(positions):
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(projection='3d')
    ax.set_title("Left cameras 3d trajectory  for {len(positions)} frames.")
    ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], s=1, c='red')

    fig.savefig(r"plots\ex3\Trajectory 3D.png")
    plt.close(fig)


def plot_both_trajectories(left_camera_positions):
    ground_truth_positions, _ = utilities.get_ground_truth_positions_and_transformations()
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory compared to ground truth of"
                 f" {len(left_camera_positions)} frames (ground truth - cyan)\n")
    ax.scatter(left_camera_positions[:, 0], left_camera_positions[:, 2], s=1, c='red')
    ax.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 2], s=1, c='cyan')

    fig.savefig(r"plots\ex3\Trajectory with ground truth 2D.png")
    plt.close(fig)


def plot_ground_truth_2d():
    ground_truth_positions, _ = utilities.get_ground_truth_positions_and_transformations()
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory compared to ground truth")
    ax.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 2], s=1, c='cyan')

    fig.savefig(r"plots\ex3\Ground truth 2D.png")
    plt.close(fig)


# -----------------------------------------------------4----------------------------------------------------------------
# 4.3
def display_track(db, track, crop=True):
    # todo: check weather the image and the xy coordinate are corresponding to each other (maybe 1 image shift?)

    frames_l, frames_r, _, _, frames_l_with_features, frames_r_with_features = \
        utilities.get_track_frames_with_and_without_features(db, track, crop=crop)

    l_vertical_concatenate = np.concatenate(frames_l_with_features, axis=0)
    r_vertical_concatenate = np.concatenate(frames_r_with_features, axis=0)

    l_r_concatenate = np.concatenate([l_vertical_concatenate, r_vertical_concatenate], axis=1)

    # cv2.imwrite(r'plots\ex4\track_r' + str(track.track_id) + '.jpg', r_vertical_concatenate)
    # cv2.imwrite(r'plots\ex4\track_l' + str(track.track_id) + '.jpg', l_vertical_concatenate)
    cv2.imwrite(r'plots\ex4\track_lr' + str(track.track_id) + '.jpg', l_r_concatenate)


# 4.4
def connectivity_graph(frames):
    """
    Present a connectivity graph: For each frame, the number of tracks outgoing to the next
    frame (the number of tracks on the frame with links also in the next frame)
    :param frames: frames
    """
    outgoings = [frame.outgoing for frame in frames]

    x = range(len(outgoings))

    f = plt.figure()
    f.set_figwidth(14)
    f.set_figheight(7)

    # plotting the points
    plt.plot(x, outgoings)

    plt.xlabel('frames')
    plt.ylabel('Outgoing tracks')
    plt.title(f'Connectivity for {FRAMES_NUM} frames')

    plt.savefig(r"plots\ex4\Connectivity_Graph.png")


# 4.5
def present_inliers_per_frame_percentage(frames):
    """
    Present a graph of the percentage of inliers per frame
    number of supporters after consensus match / stereo matches
    :return:
    """
    inliers_pers = [frame.inliers_per for frame in frames]
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_title("Inliers percentage: ")
    plt.plot(np.asarray(inliers_pers))
    plt.ylabel('Inliers percentage')
    plt.xlabel('Frame id')

    fig.savefig(r"plots\ex4\Inliers percentage.png")
    plt.close(fig)


# 4.6
def present_track_len_histogram(tracks):
    """
     Present a track length histogram graph
    :return:
    """
    track_len = [len(track) for track in tracks.values()]

    fig, ax = plt.subplots(figsize=(10, 7))
    histogram_track_len, _, _ = plt.hist(track_len)

    ax.set_title("Track length histogram")
    plt.plot(histogram_track_len)
    plt.ylabel('Track id ')
    plt.xlabel('Track lengths')

    fig.savefig(r"plots\ex4\Track length histogram.png")
    plt.close(fig)


# 4.7
def present_reprojection_error(db, track):
    frame_ids = [frame_id for frame_id in track.frames_by_ids.keys()]
    _, _, frames_l_xy, frames_r_xy, _, _ = utilities.get_track_frames_with_and_without_features(db, track)
    _, gt_trans = utilities.get_ground_truth_positions_and_transformations(seq=(frame_ids[0], frame_ids[-1] + 1))
    # _, gt_trans = utilities.get_ground_truth_positions_and_transformations(seq=(frame_ids[0], frame_ids[-1]))

    last_left_img_xy = frames_l_xy[-1]
    last_right_img_xy = frames_r_xy[-1]

    last_left_trans = gt_trans[-1]
    last_l_projection_mat = utilities.K @ last_left_trans
    last_r_projection_mat = utilities.K @ utilities.get_composition(last_left_trans, utilities.M2)
    p3d = utilities.xy_triangulation([last_left_img_xy, last_right_img_xy], last_l_projection_mat,
                                     last_r_projection_mat)

    left_projections = []
    right_projections = []

    for trans in gt_trans:
        left_proj_cam = utilities.K @ trans
        right_proj_cam = utilities.K @ utilities.get_composition(trans, utilities.M2)

        left_proj = utilities.project(p3d, left_proj_cam)
        right_proj = utilities.project(p3d, right_proj_cam)

        left_projections.append(left_proj)
        right_projections.append(right_proj)

    left_proj_dist = utilities.get_euclidean_distance(np.array(left_projections), np.array(frames_l_xy))
    right_proj_dist = utilities.get_euclidean_distance(np.array(right_projections), np.array(frames_r_xy))
    total_proj_dist = (left_proj_dist + right_proj_dist) / 2

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_title(f"Reprojection error for track: {track.track_id}")
    plt.scatter(range(len(total_proj_dist)), total_proj_dist)
    plt.ylabel('Error')
    plt.xlabel('Frames')

    fig.savefig(f"plots/ex4/reprojection_error/Reprojection error {track.track_id}.png")
    plt.close(fig)


# -----------------------------------------------------5----------------------------------------------------------------

# 5.1
def present_gtsam_re_projection_track_error(total_proj_dist, track):
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_title(f"Reprojection error for track: {track.track_id} with len = {len(track)}")
    plt.scatter(range(len(total_proj_dist)), total_proj_dist)
    plt.ylabel('Error')
    plt.xlabel('Frames')
    fig.show()
    fig.savefig(fr"plots/ex5/gtsam_reprojection_error/gtsam_reprojection_track_error {track.track_id}.png")
    plt.close(fig)


def plot_factor_re_projection_error_graph(factor_projection_errors, track):
    """
    Plots re projection error
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f"Factor Re projection error from last frame")
    plt.scatter(range(len(factor_projection_errors)), factor_projection_errors, label="Factor")
    plt.legend(loc="upper right")
    plt.ylabel('Error')
    plt.xlabel('Frames')
    fig.show()
    fig.savefig(
        fr"plots/ex5/factor_reprojection_error/Factor Re projection error graph for last frame {track.track_id}.png")
    plt.close(fig)


def plot_factor_as_func_of_re_projection_error_graph(factor_projection_errors, total_proj_dist, track):
    """
    Plots re projection error
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_title(f"Factor error as a function of a Re projection error graph for last frame")
    plt.plot(total_proj_dist, factor_projection_errors, label="Factor")
    # plt.plot(total_proj_dist, 0.5 * total_proj_dist ** 2, label="0.5x^2")
    # plt.plot(total_proj_dist, total_proj_dist ** 2, label="x^2")
    plt.legend(loc="upper left")
    plt.ylabel('Factor error')
    plt.xlabel('Re projection error')
    fig.show()
    fig.savefig(
        f"plots/ex5/factor_error_reprojection_func/"
        f"Factor error as a function of a reprojection error {track.track_id}.png")
    plt.close(fig)


def plot_left_cam_2d_trajectory(bundle_data, title=""):
    """
    Compare the left cameras relative 2d positions to the ground truth
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    cameras = [bundle_data.get_optimized_cameras_p3d()]
    cameras = utilities.gtsam_left_cameras_trajectory(cameras)
    # cameras = utilities.gtsam_left_cameras_relative_trans(cameras)
    landmarks = bundle_data.get_optimized_landmarks_p3d()

    # ax.set_title(f"{title} Left cameras and landmarks 2d trajectory of {len(cameras)} bundles")

    ax.scatter(landmarks[:, 0], landmarks[:, 2], s=1, c='orange', label="Landmarks")

    ax.scatter(cameras[:, 0], cameras[:, 2], s=1, c='red', label="Cameras")
    ax.legend(loc="upper right")
    ax.set_xlim(-200, 350)
    ax.set_ylim(-100, 500)

    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 50)

    fig.savefig(f"plots/ex5/Trajectory2D_2v/Trajectory2D_2v.png")
    plt.close(fig)


def plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth(cameras=None, landmarks=None,
                                                                       initial_estimate_poses=None, cameras_gt=None,
                                                                       title=""):
    """
    Compare the left cameras relative 2d positions to the ground truth
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_title(f"{title} Left cameras and landmarks 2d trajectory of {len(cameras)} bundles")

    if landmarks is not None:
        ax.scatter(landmarks[:, 0], landmarks[:, 2], s=1, c='orange', label="Landmarks")

    if cameras is not None:
        ax.scatter(cameras[:, 0], cameras[:, 2], s=1, c='red', label="Cameras after optimization")

    if cameras_gt is not None:
        ax.scatter(cameras_gt[:, 0], cameras_gt[:, 2], s=1, c='cyan', label="Cameras ground truth")

    if initial_estimate_poses is not None:
        ax.scatter(initial_estimate_poses[:, 0], initial_estimate_poses[:, 2], s=1, c='purple', label="Initial estimate")

    ax.legend(loc="upper right")
    ax.set_xlim(-350, 350)
    ax.set_ylim(-200, 500)

    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 50)

    fig.savefig(f"plots/ex5/FullTrajectory2D/FullTrajectory2D.png")
    plt.close(fig)
