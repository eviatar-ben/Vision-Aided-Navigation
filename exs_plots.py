import matplotlib.pyplot as plt
import cv2
import numpy as np
from ex4 import FRAMES_NUM

GROUND_TRUTH_PATH = r"../dataset/poses/00.txt"
DATA_PATH = r'C:/Users/eviatar/Desktop/eviatar/Study/YearD/semester b/VAN/VAN_ex/dataset/sequences/00/'
FIRST_IMAGE = 000000


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
    def get_ground_truth_transformations2(left_cam_trans_path=GROUND_TRUTH_PATH):
        def relative(t):
            return -1 * t[:, :3].T @ t[:, 3]

        ground_truth_trans = []
        with open(left_cam_trans_path) as f:
            lines = f.readlines()
        for i in range(3450):
            left_mat = np.array(lines[i].split(" "))[:-1].astype(float).reshape((3, 4))
            ground_truth_trans.append(left_mat)

        relative_cameras_pos_arr = []
        for t in ground_truth_trans:
            relative_cameras_pos_arr.append(relative(t))
        return np.array(relative_cameras_pos_arr)

    ground_truth_positions = get_ground_truth_transformations2()
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory compared to ground truth of"
                 f" {len(left_camera_positions)} frames (ground truth - cyan)\n")
    ax.scatter(left_camera_positions[:, 0], left_camera_positions[:, 2], s=1, c='red')
    ax.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 2], s=1, c='cyan')

    fig.savefig(r"plots\ex3\Trajectory with ground truth 2D.png")
    plt.close(fig)


def plot_ground_truth_2d():
    def get_ground_truth_transformations2(left_cam_trans_path=GROUND_TRUTH_PATH):
        def relative(t):
            return -1 * t[:, :3].T @ t[:, 3]

        ground_truth_trans = []
        with open(left_cam_trans_path) as f:
            lines = f.readlines()
        for i in range(3450):
            left_mat = np.array(lines[i].split(" "))[:-1].astype(float).reshape((3, 4))
            ground_truth_trans.append(left_mat)

        relative_cameras_pos_arr = []
        for t in ground_truth_trans:
            relative_cameras_pos_arr.append(relative(t))
        return np.array(relative_cameras_pos_arr)

    ground_truth_positions = get_ground_truth_transformations2()
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory compared to ground truth")
    ax.scatter(ground_truth_positions[:, 0], ground_truth_positions[:, 2], s=1, c='cyan')

    fig.savefig(r"plots\ex3\Ground truth 2D.png")
    plt.close(fig)


# -----------------------------------------------------4----------------------------------------------------------------
def display_track(db, track):
    frame_ids = [frame_id for frame_id in track.frames_by_ids.keys()]
    frames_path = ['{:06d}.png'.format(FIRST_IMAGE + frame.frame_id) for frame in track.frames_by_ids.values()]
    frames_l = [cv2.imread(DATA_PATH + 'image_0/' + frame_path) for frame_path in frames_path]
    frames_r = [cv2.imread(DATA_PATH + 'image_1/' + frame_path) for frame_path in frames_path]

    frames_l_xy = []
    for frame_id in frame_ids:
        frames_l_xy.append((db.get_feature_location(frame_id, track.track_id)[0],
                            db.get_feature_location(frame_id, track.track_id)[2]))

    frames_r_xy = []
    for frame_id in frame_ids:
        frames_r_xy.append((db.get_feature_location(frame_id, track.track_id)[1],
                            db.get_feature_location(frame_id, track.track_id)[2]))

    # todo check if its possible to avoid truncate the coordinates
    frames_l_with_features = [cv2.circle(frame, (int(xy[0]), int(xy[1])), 1, (255, 0, 0), 5) for frame, xy in
                              zip(frames_l, frames_l_xy)]
    l_vertical_concatenate = np.concatenate(frames_l_with_features, axis=0)

    frames_r_with_features = [cv2.circle(frame, (int(xy[0]), int(xy[1])), 1, (255, 0, 0), 5) for frame, xy in
                              zip(frames_r, frames_r_xy)]
    r_vertical_concatenate = np.concatenate(frames_r_with_features, axis=0)

    l_r_concatenate = np.concatenate([l_vertical_concatenate, r_vertical_concatenate], axis=1)

    cv2.imwrite(r'plots\ex4\track_r' + str(track.track_id) + '.jpg', r_vertical_concatenate)
    cv2.imwrite(r'plots\ex4\track_l' + str(track.track_id) + '.jpg', l_vertical_concatenate)
    cv2.imwrite(r'plots\ex4\track_lr' + str(track.track_id) + '.jpg', l_r_concatenate)


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


def present_inliers_per_frame_percentage():
    """
    Present a graph of the percentage of inliers per frame
    :return:
    """
    pass


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
