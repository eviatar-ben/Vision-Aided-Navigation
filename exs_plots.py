import matplotlib.pyplot as plt
import cv2
import numpy as np


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
    cv2.imwrite("mutual_key_points.jpg", img_kp1)


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


# 3.5:


def plot_clouds(p3d, transform_p3d):
    # fig = px.scatter_3d(p3d, x=0, y=1, z=2, labels={
    #     '0': "X axis",
    #     '1': "Y axis",
    #     '2': "Z axis"},
    #                     title=f'')
    # fig.show()
    # fig = px.scatter_3d(transform_p3d, x=0, y=1, z=2, labels={
    #     '0': "X axis",
    #     '1': "Y axis",
    #     '2': "Z axis"},
    #                     title=f'')
    # fig.show()
    pass
# 3.6:
def plot_positions(positions):
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_title(f"Left cameras 2d trajectory for {len(positions)} frames.")
    ax.scatter(positions[:, 0], positions[:, 2], s=1, c='red')

    fig.savefig(f"Trajectory_from_cmd.png")
    plt.close(fig)
