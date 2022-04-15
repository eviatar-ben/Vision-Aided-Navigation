import matplotlib.pyplot as plt
import cv2


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
