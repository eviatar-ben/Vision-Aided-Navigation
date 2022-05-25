import cv2
import numpy as np

GROUND_TRUTH_PATH = r"../dataset/poses/00.txt"
DATA_PATH = r'../dataset/sequences/00/'
FIRST_IMAGE = 000000
IMAGE_WIDTH = 1241
IMAGE_HEIGHT = 376
FRAMES_NUM = 3450


# -----------------------------------------------------2---------------------------------------------------------------


def read_cameras():
    with open(DATA_PATH + 'calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
        l1 = [float(i) for i in l1]
        m1 = np.array(l1).reshape(3, 4)
        l2 = [float(i) for i in l2]
        m2 = np.array(l2).reshape(3, 4)
        k = m1[:, :3]
        m1 = np.linalg.inv(k) @ m1
        m2 = np.linalg.inv(k) @ m2
        return k, m1, m2


def xy_triangulation(in_liers, m1c, m2c):
    """
    triangulation for case where: in_lier is xy point.
    (the others are for inliers as key points).
    """
    ps = cv2.triangulatePoints(m1c, m2c, np.array(in_liers[0]).T, np.array(in_liers[1]).T).T
    return np.squeeze(cv2.convertPointsFromHomogeneous(ps))


def triangulation(in_liers, m1c, m2c):
    def compute_homogenates_matrix(p, q):
        p, q = p.pt, q.pt
        return np.array([p[0] * m1c[2] - m1c[0],
                         p[1] * m1c[2] - m1c[1],
                         q[0] * m2c[2] - m2c[0],
                         q[1] * m2c[2] - m2c[1]])

    xs = []
    for i, j in zip(in_liers[0], in_liers[1]):
        homogenates_matrix = compute_homogenates_matrix(i, j)
        _, _, vh = np.linalg.svd(homogenates_matrix)
        last_col = vh[-1]
        xs.append(last_col[:3] / last_col[3])
    return np.array(xs)


def cv2_triangulation(in_liers, m1c, m2c):
    def edit_to_pt():
        for i, j in zip(in_liers[0], in_liers[1]):
            points1.append(i.pt)
            points2.append(j.pt)

    points1 = []
    points2 = []
    edit_to_pt()

    ps = cv2.triangulatePoints(m1c, m2c, np.array(points1).T, np.array(points2).T).T
    return np.squeeze(cv2.convertPointsFromHomogeneous(ps))


def project(p3d_pts, projection_cam_mat):
    hom_projected = p3d_pts @ projection_cam_mat[:, :3].T + projection_cam_mat[:, 3].T
    projected = hom_projected[:2] / hom_projected[2]
    return projected


# -----------------------------------------------------3----------------------------------------------------------------
K, M1, M2 = read_cameras()


def get_euclidean_distance(a, b, ):
    # supporters are determined by norma 2.:
    distances = np.sqrt(np.power(a - b, 2).sum(axis=1))
    # np.linalg.norm(a - b)
    return distances


def get_ground_truth_positions_and_transformations(left_cam_trans_path=GROUND_TRUTH_PATH, seq=(0, FRAMES_NUM)):
    def relative(t):
        return -1 * t[:, :3].T @ t[:, 3]

    ground_truth_trans = []
    with open(left_cam_trans_path) as f:
        lines = f.readlines()
    # for i in range(3450):
    for i in range(seq[0], seq[1]):
        left_mat = np.array(lines[i].split(" "))[:-1].astype(float).reshape((3, 4))
        ground_truth_trans.append(left_mat)

    relative_cameras_pos_arr = []
    for t in ground_truth_trans:
        relative_cameras_pos_arr.append(relative(t))
    return np.array(relative_cameras_pos_arr), ground_truth_trans


def get_composition(trans1, trans2):
    """
    trans1 A -> B
    trans2 : B -> C
    :return mat : A -> C
    """
    # R2R1 * v + R2t1 + t2.
    r2r1 = trans2[:, :-1] @ (trans1[:, :-1])
    r2t1_t2 = (trans2[:, :-1]) @ (trans1[:, -1]) + trans2[:, -1]
    ext_r1 = np.column_stack((r2r1, r2t1_t2))
    return ext_r1


# -----------------------------------------------------4----------------------------------------------------------------

def get_track_in_len(db, track_min_len, rand=False):
    import random
    memo = [83001, 4833, 56640]
    track = None
    found = False
    tracks = list(db.tracks.values())
    if not rand:
        for track in tracks:
            if len(track) >= track_min_len:
                return track
    else:
        while not found:
            idx = random.randint(0, len(tracks) - 1)
            if len(tracks[idx]) == track_min_len:
                track = tracks[idx]
                found = True
    # return tracks[2421]
    return track


def crop_image(xy, img, crop_size):
    """
    Crops image "img" to size of "crop_size" X "crop_size" around the coordinates "xy"
    :return: Cropped image
    """
    r_x = int(min(IMAGE_WIDTH, xy[0] + crop_size))
    l_x = int(max(0, xy[0] - crop_size))
    u_y = int(max(0, xy[1] - crop_size))
    d_y = int(min(IMAGE_HEIGHT, xy[1] + crop_size))

    return img[u_y: d_y, l_x: r_x]


def get_track_frames_with_and_without_features(db, track, crop=False):
    """
    given database and track the function will return an array of images (frames)
    of the corresponding track both left and right frames,
    and both frames with and without the features loaded in the frames
    :param db:
    :param track:
    :return:
    """
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

    # todo check if its possible to avoid rounding the coordinates
    frames_l_with_features = [cv2.circle(frame, (int(round(xy[0])), int(round(xy[1]))), 1, (0, 0, 255), 5) for frame, xy
                              in zip(frames_l, frames_l_xy)]

    frames_r_with_features = [cv2.circle(frame, (int(round(xy[0])), int(round(xy[1]))), 1, (0, 0, 255), 5) for frame, xy
                              in zip(frames_r, frames_r_xy)]
    if crop:
        frames_l_with_features = [crop_image(xy, image, 100) for xy, image in
                                  zip(frames_l_xy, frames_l_with_features)]
        frames_r_with_features = [crop_image(xy, image, 100) for xy, image in
                                  zip(frames_r_xy, frames_r_with_features)]

    return frames_l, frames_r, frames_l_xy, frames_r_xy, frames_l_with_features, frames_r_with_features


def compute_reprojection_square_dist(img_projected_pts, img_pts_coor):
    """
    Check the euclidean dist between the projected points and correspond pixel locations
    :param img_projected_pts:
    :param img_pts_coor:
    :return:
    """
    img_pts_diff = img_projected_pts - img_pts_coor  # (x1, y1), (x2, y2) -> (x1 - x2, y1 - y2)
    left0_squared_dist = np.einsum("ij,ij->i", img_pts_diff, img_pts_diff)  # (x1 - x2)^2 + (y1 - y2)^2
    return left0_squared_dist


def compute_reprojection_euclidean_dist(img_projected_pts, img_pts_coor):
    left0_squared_dist = compute_reprojection_square_dist(img_projected_pts, img_pts_coor)
    return np.sqrt(left0_squared_dist)


# -----------------------------------------------------5----------------------------------------------------------------
def present_factor_error_differences(factor_error_after_optimization, factor_error_before_optimization):
    print("First Bundle Errors:")
    print("Error before optimization: ", factor_error_before_optimization)
    print("Error after optimization: ", factor_error_after_optimization)


def reverse_ext(ext):
    """
    this function gets an extrinsic matrix  world coord -> camera coord
    and return the "opposite" camera coord -> world coord
    """
    R = ext[:, :3]
    t = ext[:, 3]

    rev_R = R.T
    # rev_t = -rev_R @ t
    rev_t = -R.T @ t

    return np.hstack((rev_R, rev_t.reshape(3, 1)))


def get_track_frames_with_features(db, track):
    """
    given database and track the function will return 2 arrays of the track's feature left and right
    (each "side" in different array)
    :param db:
    :param track:
    :return:
    """
    frame_ids = [frame_id for frame_id in track.frames_by_ids.keys()]
    frames_l_xy = []
    frames_r_xy = []

    for frame_id in frame_ids:
        features_coord = db.get_feature_location(frame_id, track.track_id)
        frames_l_xy.append((features_coord[0],
                            features_coord[2]))
        frames_r_xy.append((features_coord[1],
                            features_coord[2]))

    return frames_l_xy, frames_r_xy


def compose_transformations(first_ex_mat, second_ex_mat):
    """
    Compute the composition of two extrinsic camera matrices.
    first_cam : A -> B
    second_cam : B -> C
    composed mat : A -> C
    """
    # [R2 | t2] @ [ R1 | t1] = [R2 @ R1 | R2 @ t1 + t2]
    #             [000 | 1 ]
    hom1 = np.append(first_ex_mat, [np.array([0, 0, 0, 1])], axis=0)
    return second_ex_mat @ hom1


def compute_square_dist(pts_lst1, pts_lst2, dim="3d"):
    """
    Check the euclidean dist between the projected d2_points and correspond pixel locations
    :param pts_lst1:
    :param pts_lst2:
    :return:
    """
    pts_sub = pts_lst1 - pts_lst2  # (x1, y1), (x2, y2) -> (x1 - x2, y1 - y2)
    if dim == "2d":
        squared_dist = np.einsum("ij,ij->i", pts_sub, pts_sub)  # (x1 - x2)^2 + (y1 - y2)^2
    elif dim == "3d":
        squared_dist = np.linalg.norm(pts_sub, axis=1)
    return squared_dist


def euclidean_dist(pts_lst1, pts_lst2, dim="3d"):
    squared_dist = compute_square_dist(pts_lst1, pts_lst2, dim=dim)
    return np.sqrt(squared_dist)


def get_projection_factors_errors(factors, values):
    errors = []
    for factor in factors:
        errors.append(factor.error(values))

    return np.array(errors)


def get_total_projection_error(left_projections, right_projections, left_locations, right_locations):
    left_proj_dist = euclidean_dist(np.array(left_projections), np.array(left_locations))
    right_proj_dist = euclidean_dist(np.array(right_projections), np.array(right_locations))
    total_proj_dist = (left_proj_dist + right_proj_dist) / 2
    return total_proj_dist


def gtsam_left_cameras_trajectory(relative_T_arr):
    """
    Computes the left cameras 3d positions relative to the starting position
    :param T_arr: relative to first camera transformations array
    :return: numpy array with dimension num T_arr X 3
    """
    relative_cameras_pos_arr = []
    for t in relative_T_arr:
        relative_cameras_pos_arr.append(t)
    return np.array(relative_cameras_pos_arr)


# problems with
# 150 155
# 2845 2850
# 3195 3200
fives = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55),
         (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 95), (95, 100), (100, 105),
         (105, 110), (110, 115), (115, 120), (120, 125), (125, 130), (130, 135), (135, 140),
         (140, 148), (148, 152), (152, 160),
         (160, 165), (165, 170), (170, 175), (175, 180), (180, 185), (185, 190), (190, 195),
         (195, 200), (200, 205), (205, 210), (210, 215), (215, 220), (220, 225), (225, 230), (230, 235), (235, 240),
         (240, 245), (245, 250), (250, 255), (255, 260), (260, 265), (265, 270), (270, 275), (275, 280), (280, 285),
         (285, 290), (290, 295), (295, 300), (300, 305), (305, 310), (310, 315), (315, 320), (320, 325), (325, 330),
         (330, 335), (335, 340), (340, 345), (345, 350), (350, 355), (355, 360), (360, 365), (365, 370), (370, 375),
         (375, 380), (380, 385), (385, 390), (390, 395), (395, 400), (400, 405), (405, 410), (410, 415), (415, 420),
         (420, 425), (425, 430), (430, 435), (435, 440), (440, 445), (445, 450), (450, 455), (455, 460), (460, 465),
         (465, 470), (470, 475), (475, 480), (480, 485), (485, 490), (490, 495), (495, 500), (500, 505), (505, 510),
         (510, 515), (515, 520), (520, 525), (525, 530), (530, 535), (535, 540), (540, 545), (545, 550), (550, 555),
         (555, 560), (560, 565), (565, 570), (570, 575), (575, 580), (580, 585), (585, 590), (590, 595), (595, 600),
         (600, 605), (605, 610), (610, 615), (615, 620), (620, 625), (625, 630), (630, 635), (635, 640), (640, 645),
         (645, 650), (650, 655), (655, 660), (660, 665), (665, 670), (670, 675), (675, 680), (680, 685), (685, 690),
         (690, 695), (695, 700), (700, 705), (705, 710), (710, 715), (715, 720), (720, 725), (725, 730), (730, 735),
         (735, 740), (740, 745), (745, 750), (750, 755), (755, 760), (760, 765), (765, 770), (770, 775), (775, 780),
         (780, 785), (785, 790), (790, 795), (795, 800), (800, 805), (805, 810), (810, 815), (815, 820), (820, 825),
         (825, 830), (830, 835), (835, 840), (840, 845), (845, 850), (850, 855), (855, 860), (860, 865), (865, 870),
         (870, 875), (875, 880), (880, 885), (885, 890), (890, 895), (895, 900), (900, 905), (905, 910), (910, 915),
         (915, 920), (920, 925), (925, 930), (930, 935), (935, 940), (940, 945), (945, 950), (950, 955), (955, 960),
         (960, 965), (965, 970), (970, 975), (975, 980), (980, 985), (985, 990), (990, 995), (995, 1000), (1000, 1005),
         (1005, 1010), (1010, 1015), (1015, 1020), (1020, 1025), (1025, 1030), (1030, 1035), (1035, 1040), (1040, 1045),
         (1045, 1050), (1050, 1055), (1055, 1060), (1060, 1065), (1065, 1070), (1070, 1075), (1075, 1080), (1080, 1085),
         (1085, 1090), (1090, 1095), (1095, 1100), (1100, 1105), (1105, 1110), (1110, 1115), (1115, 1120), (1120, 1125),
         (1125, 1130), (1130, 1135), (1135, 1140), (1140, 1145), (1145, 1150), (1150, 1155), (1155, 1160), (1160, 1165),
         (1165, 1170), (1170, 1175), (1175, 1180), (1180, 1185), (1185, 1190), (1190, 1195), (1195, 1200), (1200, 1205),
         (1205, 1210), (1210, 1215), (1215, 1220), (1220, 1225), (1225, 1230), (1230, 1235), (1235, 1240), (1240, 1245),
         (1245, 1250), (1250, 1255), (1255, 1260), (1260, 1265), (1265, 1270), (1270, 1275), (1275, 1280), (1280, 1285),
         (1285, 1290), (1290, 1295), (1295, 1300), (1300, 1305), (1305, 1310), (1310, 1315), (1315, 1320), (1320, 1325),
         (1325, 1330), (1330, 1335), (1335, 1340), (1340, 1345), (1345, 1350), (1350, 1355), (1355, 1360), (1360, 1365),
         (1365, 1370), (1370, 1375), (1375, 1380), (1380, 1385), (1385, 1390), (1390, 1395), (1395, 1400), (1400, 1405),
         (1405, 1410), (1410, 1415), (1415, 1420), (1420, 1425), (1425, 1430), (1430, 1435), (1435, 1440), (1440, 1445),
         (1445, 1450), (1450, 1455), (1455, 1460), (1460, 1465), (1465, 1470), (1470, 1475), (1475, 1480), (1480, 1485),
         (1485, 1490), (1490, 1495), (1495, 1500), (1500, 1505), (1505, 1510), (1510, 1515), (1515, 1520), (1520, 1525),
         (1525, 1530), (1530, 1535), (1535, 1540), (1540, 1545), (1545, 1550), (1550, 1555), (1555, 1560), (1560, 1565),
         (1565, 1570), (1570, 1575), (1575, 1580), (1580, 1585), (1585, 1590), (1590, 1595), (1595, 1600), (1600, 1605),
         (1605, 1610), (1610, 1615), (1615, 1620), (1620, 1625), (1625, 1630), (1630, 1635), (1635, 1640), (1640, 1645),
         (1645, 1650), (1650, 1655), (1655, 1660), (1660, 1665), (1665, 1670), (1670, 1675), (1675, 1680), (1680, 1685),
         (1685, 1690), (1690, 1695), (1695, 1700), (1700, 1705), (1705, 1710), (1710, 1715), (1715, 1720), (1720, 1725),
         (1725, 1730), (1730, 1735), (1735, 1740), (1740, 1745), (1745, 1750), (1750, 1755), (1755, 1760), (1760, 1765),
         (1765, 1770), (1770, 1775), (1775, 1780), (1780, 1785), (1785, 1790), (1790, 1795), (1795, 1800), (1800, 1805),
         (1805, 1810), (1810, 1815), (1815, 1820), (1820, 1825), (1825, 1830), (1830, 1835), (1835, 1840), (1840, 1845),
         (1845, 1850), (1850, 1855), (1855, 1860), (1860, 1865), (1865, 1870), (1870, 1875), (1875, 1880), (1880, 1885),
         (1885, 1890), (1890, 1895), (1895, 1900), (1900, 1905), (1905, 1910), (1910, 1915), (1915, 1920), (1920, 1925),
         (1925, 1930), (1930, 1935), (1935, 1940), (1940, 1945), (1945, 1950), (1950, 1955), (1955, 1960), (1960, 1965),
         (1965, 1970), (1970, 1975), (1975, 1980), (1980, 1985), (1985, 1990), (1990, 1995), (1995, 2000), (2000, 2005),
         (2005, 2010), (2010, 2015), (2015, 2020), (2020, 2025), (2025, 2030), (2030, 2035), (2035, 2040), (2040, 2045),
         (2045, 2050), (2050, 2055), (2055, 2060), (2060, 2065), (2065, 2070), (2070, 2075), (2075, 2080), (2080, 2085),
         (2085, 2090), (2090, 2095), (2095, 2100), (2100, 2105), (2105, 2110), (2110, 2115), (2115, 2120), (2120, 2125),
         (2125, 2130), (2130, 2135), (2135, 2140), (2140, 2145), (2145, 2150), (2150, 2155), (2155, 2160), (2160, 2165),
         (2165, 2170), (2170, 2175), (2175, 2180), (2180, 2185), (2185, 2190), (2190, 2195), (2195, 2200), (2200, 2205),
         (2205, 2210), (2210, 2215), (2215, 2220), (2220, 2225), (2225, 2230), (2230, 2235), (2235, 2240), (2240, 2245),
         (2245, 2250), (2250, 2255), (2255, 2260), (2260, 2265), (2265, 2270), (2270, 2275), (2275, 2280), (2280, 2285),
         (2285, 2290), (2290, 2295), (2295, 2300), (2300, 2305), (2305, 2310), (2310, 2315), (2315, 2320), (2320, 2325),
         (2325, 2330), (2330, 2335), (2335, 2340), (2340, 2345), (2345, 2350), (2350, 2355), (2355, 2360), (2360, 2365),
         (2365, 2370), (2370, 2375), (2375, 2380), (2380, 2385), (2385, 2390), (2390, 2395), (2395, 2400), (2400, 2405),
         (2405, 2410), (2410, 2415), (2415, 2420), (2420, 2425), (2425, 2430), (2430, 2435), (2435, 2440), (2440, 2445),
         (2445, 2450), (2450, 2455), (2455, 2460), (2460, 2465), (2465, 2470), (2470, 2475), (2475, 2480), (2480, 2485),
         (2485, 2490), (2490, 2495), (2495, 2500), (2500, 2505), (2505, 2510), (2510, 2515), (2515, 2520), (2520, 2525),
         (2525, 2530), (2530, 2535), (2535, 2540), (2540, 2545), (2545, 2550), (2550, 2555), (2555, 2560), (2560, 2565),
         (2565, 2570), (2570, 2575), (2575, 2580), (2580, 2585), (2585, 2590), (2590, 2595), (2595, 2600), (2600, 2605),
         (2605, 2610), (2610, 2615), (2615, 2620), (2620, 2625), (2625, 2630), (2630, 2635), (2635, 2640), (2640, 2645),
         (2645, 2650), (2650, 2655), (2655, 2660), (2660, 2665), (2665, 2670), (2670, 2675), (2675, 2680), (2680, 2685),
         (2685, 2690), (2690, 2695), (2695, 2700), (2700, 2705), (2705, 2710), (2710, 2715), (2715, 2720), (2720, 2725),
         (2725, 2730), (2730, 2735), (2735, 2740), (2740, 2745), (2745, 2750), (2750, 2755), (2755, 2760), (2760, 2765),
         (2765, 2770), (2770, 2775), (2775, 2780), (2780, 2785), (2785, 2790), (2790, 2795), (2795, 2800), (2800, 2805),
         (2805, 2810), (2810, 2815), (2815, 2820), (2820, 2825), (2825, 2830), (2830, 2835), (2835, 2840), (2840, 2845),
         (2845, 2847), (2847, 2850), (2850, 2855),
         (2855, 2860), (2860, 2865), (2865, 2870), (2870, 2875), (2875, 2880), (2880, 2885),
         (2885, 2890), (2890, 2895), (2895, 2900), (2900, 2905), (2905, 2910), (2910, 2915), (2915, 2920), (2920, 2925),
         (2925, 2930), (2930, 2935), (2935, 2940), (2940, 2945), (2945, 2950), (2950, 2955), (2955, 2960), (2960, 2965),
         (2965, 2970), (2970, 2975), (2975, 2980), (2980, 2985), (2985, 2990), (2990, 2995), (2995, 3000), (3000, 3005),
         (3005, 3010), (3010, 3015), (3015, 3020), (3020, 3025), (3025, 3030), (3030, 3035), (3035, 3040), (3040, 3045),
         (3045, 3050), (3050, 3055), (3055, 3060), (3060, 3065), (3065, 3070), (3070, 3075), (3075, 3080), (3080, 3085),
         (3085, 3090), (3090, 3095), (3095, 3100), (3100, 3105), (3105, 3110), (3110, 3115), (3115, 3120), (3120, 3125),
         (3125, 3130), (3130, 3135), (3135, 3140), (3140, 3145), (3145, 3150), (3150, 3155), (3155, 3160), (3160, 3165),
         (3165, 3170), (3170, 3175), (3175, 3180), (3180, 3185), (3185, 3190), (3190, 3195),
         (3195, 3197), (3197, 3200),
         (3200, 3205), (3205, 3210), (3210, 3215), (3215, 3220), (3220, 3225), (3225, 3230), (3230, 3235), (3235, 3240),
         (3240, 3245), (3245, 3250), (3250, 3255), (3255, 3260), (3260, 3265), (3265, 3270), (3270, 3275), (3275, 3280),
         (3280, 3285), (3285, 3290), (3290, 3295), (3295, 3300), (3300, 3305), (3305, 3310), (3310, 3315), (3315, 3320),
         (3320, 3325), (3325, 3330), (3330, 3335), (3335, 3340), (3340, 3345), (3345, 3350), (3350, 3355), (3355, 3360),
         (3360, 3365), (3365, 3370), (3370, 3375), (3375, 3380), (3380, 3385), (3385, 3390), (3390, 3395), (3395, 3400),
         (3400, 3405), (3405, 3410), (3410, 3415), (3415, 3420), (3420, 3425), (3425, 3430), (3430, 3435), (3435, 3440),
         (3440, 3445), (3445, 3449)]
tens = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 110),
        (110, 120), (120, 130), (130, 140), (140, 150), (150, 160), (160, 170), (170, 180), (180, 190), (190, 200),
        (200, 210), (210, 220), (220, 230), (230, 240), (240, 250), (250, 260), (260, 270), (270, 280), (280, 290),
        (290, 300), (300, 310), (310, 320), (320, 330), (330, 340), (340, 350), (350, 360), (360, 370), (370, 380),
        (380, 390), (390, 400), (400, 410), (410, 420), (420, 430), (430, 440), (440, 450), (450, 460), (460, 470),
        (470, 480), (480, 490), (490, 500), (500, 510), (510, 520), (520, 530), (530, 540), (540, 550), (550, 560),
        (560, 570), (570, 580), (580, 590), (590, 600), (600, 610), (610, 620), (620, 630), (630, 640), (640, 650),
        (650, 660), (660, 670), (670, 680), (680, 690), (690, 700), (700, 710), (710, 720), (720, 730), (730, 740),
        (740, 750), (750, 760), (760, 770), (770, 780), (780, 790), (790, 800), (800, 810), (810, 820), (820, 830),
        (830, 840), (840, 850), (850, 860), (860, 870), (870, 880), (880, 890), (890, 900), (900, 910), (910, 920),
        (920, 930), (930, 940), (940, 950), (950, 960), (960, 970), (970, 980), (980, 990), (990, 1000), (1000, 1010),
        (1010, 1020), (1020, 1030), (1030, 1040), (1040, 1050), (1050, 1060), (1060, 1070), (1070, 1080), (1080, 1090),
        (1090, 1100), (1100, 1110), (1110, 1120), (1120, 1130), (1130, 1140), (1140, 1150), (1150, 1160), (1160, 1170),
        (1170, 1180), (1180, 1190), (1190, 1200), (1200, 1210), (1210, 1220), (1220, 1230), (1230, 1240), (1240, 1250),
        (1250, 1260), (1260, 1270), (1270, 1280), (1280, 1290), (1290, 1300), (1300, 1310), (1310, 1320), (1320, 1330),
        (1330, 1340), (1340, 1350), (1350, 1360), (1360, 1370), (1370, 1380), (1380, 1390), (1390, 1400), (1400, 1410),
        (1410, 1420), (1420, 1430), (1430, 1440), (1440, 1450), (1450, 1460), (1460, 1470), (1470, 1480), (1480, 1490),
        (1490, 1500), (1500, 1510), (1510, 1520), (1520, 1530), (1530, 1540), (1540, 1550), (1550, 1560), (1560, 1570),
        (1570, 1580), (1580, 1590), (1590, 1600), (1600, 1610), (1610, 1620), (1620, 1630), (1630, 1640), (1640, 1650),
        (1650, 1660), (1660, 1670), (1670, 1680), (1680, 1690), (1690, 1700), (1700, 1710), (1710, 1720), (1720, 1730),
        (1730, 1740), (1740, 1750), (1750, 1760), (1760, 1770), (1770, 1780), (1780, 1790), (1790, 1800), (1800, 1810),
        (1810, 1820), (1820, 1830), (1830, 1840), (1840, 1850), (1850, 1860), (1860, 1870), (1870, 1880), (1880, 1890),
        (1890, 1900), (1900, 1910), (1910, 1920), (1920, 1930), (1930, 1940), (1940, 1950), (1950, 1960), (1960, 1970),
        (1970, 1980), (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2020), (2020, 2030), (2030, 2040), (2040, 2050),
        (2050, 2060), (2060, 2070), (2070, 2080), (2080, 2090), (2090, 2100), (2100, 2110), (2110, 2120), (2120, 2130),
        (2130, 2140), (2140, 2150), (2150, 2160), (2160, 2170), (2170, 2180), (2180, 2190), (2190, 2200), (2200, 2210),
        (2210, 2220), (2220, 2230), (2230, 2240), (2240, 2250), (2250, 2260), (2260, 2270), (2270, 2280), (2280, 2290),
        (2290, 2300), (2300, 2310), (2310, 2320), (2320, 2330), (2330, 2340), (2340, 2350), (2350, 2360), (2360, 2370),
        (2370, 2380), (2380, 2390), (2390, 2400), (2400, 2410), (2410, 2420), (2420, 2430), (2430, 2440), (2440, 2450),
        (2450, 2460), (2460, 2470), (2470, 2480), (2480, 2490), (2490, 2500), (2500, 2510), (2510, 2520), (2520, 2530),
        (2530, 2540), (2540, 2550), (2550, 2560), (2560, 2570), (2570, 2580), (2580, 2590), (2590, 2600), (2600, 2610),
        (2610, 2620), (2620, 2630), (2630, 2640), (2640, 2650), (2650, 2660), (2660, 2670), (2670, 2680), (2680, 2690),
        (2690, 2700), (2700, 2710), (2710, 2720), (2720, 2730), (2730, 2740), (2740, 2750), (2750, 2760), (2760, 2770),
        (2770, 2780), (2780, 2790), (2790, 2800), (2800, 2810), (2810, 2820), (2820, 2830), (2830, 2840), (2840, 2850),
        (2850, 2860), (2860, 2870), (2870, 2880), (2880, 2890), (2890, 2900), (2900, 2910), (2910, 2920), (2920, 2930),
        (2930, 2940), (2940, 2950), (2950, 2960), (2960, 2970), (2970, 2980), (2980, 2990), (2990, 3000), (3000, 3010),
        (3010, 3020), (3020, 3030), (3030, 3040), (3040, 3050), (3050, 3060), (3060, 3070), (3070, 3080), (3080, 3090),
        (3090, 3100), (3100, 3110), (3110, 3120), (3120, 3130), (3130, 3140), (3140, 3150), (3150, 3160), (3160, 3170),
        (3170, 3180), (3180, 3190), (3190, 3200), (3200, 3210), (3210, 3220), (3220, 3230), (3230, 3240), (3240, 3250),
        (3250, 3260), (3260, 3270), (3270, 3280), (3280, 3290), (3290, 3300), (3300, 3310), (3310, 3320), (3320, 3330),
        (3330, 3340), (3340, 3350), (3350, 3360), (3360, 3370), (3370, 3380), (3380, 3390), (3390, 3400), (3400, 3410),
        (3410, 3420), (3420, 3430), (3430, 3440)]

tens2 = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 110),
         (110, 120), (120, 130), (130, 140), (140, 148), (148, 152), (152, 160), (160, 170), (170, 180), (180, 190),
         (190, 200),
         (200, 210), (210, 220), (220, 230), (230, 240), (240, 250), (250, 260), (260, 270), (270, 280), (280, 290),
         (290, 300), (300, 310), (310, 320), (320, 330), (330, 340), (340, 350), (350, 360), (360, 370), (370, 380),
         (380, 390), (390, 400), (400, 410), (410, 420), (420, 430), (430, 440), (440, 450), (450, 460), (460, 470),
         (470, 480), (480, 490), (490, 500), (500, 510), (510, 520), (520, 530), (530, 540), (540, 550), (550, 560),
         (560, 570), (570, 580), (580, 590), (590, 600), (600, 610), (610, 620), (620, 630), (630, 640), (640, 650),
         (650, 660), (660, 670), (670, 680), (680, 690), (690, 700), (700, 710), (710, 720), (720, 730), (730, 740),
         (740, 750), (750, 760), (760, 770), (770, 780), (780, 790), (790, 800), (800, 810), (810, 820), (820, 830),
         (830, 835),
         (835, 844), (844, 852), (852, 860), (860, 870), (870, 880), (880, 890), (890, 900), (900, 905), (905, 910),
         (910, 920),
         (920, 930), (930, 940), (940, 950), (950, 960), (960, 970), (970, 980), (980, 990), (990, 1000), (1000, 1010),
         (1010, 1020), (1020, 1030), (1030, 1040), (1040, 1050), (1050, 1060), (1060, 1070), (1070, 1080), (1080, 1090),
         (1090, 1100), (1100, 1110), (1110, 1120), (1120, 1130), (1130, 1140), (1140, 1150), (1150, 1160), (1160, 1170),
         (1170, 1180), (1180, 1190), (1190, 1200), (1200, 1210), (1210, 1220), (1220, 1230), (1230, 1240), (1240, 1250),
         (1250, 1260), (1260, 1270), (1270, 1280), (1280, 1290), (1290, 1300), (1300, 1310), (1310, 1320), (1320, 1330),
         (1330, 1340), (1340, 1350), (1350, 1360), (1360, 1370), (1370, 1380), (1380, 1390), (1390, 1400), (1400, 1410),
         (1410, 1420), (1420, 1430), (1430, 1440), (1440, 1450), (1450, 1460), (1460, 1470), (1470, 1480), (1480, 1490),
         (1490, 1500), (1500, 1510), (1510, 1520), (1520, 1530), (1530, 1540), (1540, 1550), (1550, 1560), (1560, 1570),
         (1570, 1580), (1580, 1590), (1590, 1600), (1600, 1610), (1610, 1620), (1620, 1630), (1630, 1640), (1640, 1650),
         (1650, 1660), (1660, 1670), (1670, 1680), (1680, 1690), (1690, 1700), (1700, 1710), (1710, 1720), (1720, 1730),
         (1730, 1740), (1740, 1750), (1750, 1760), (1760, 1770), (1770, 1780), (1780, 1790), (1790, 1800), (1800, 1810),
         (1810, 1820), (1820, 1830), (1830, 1840), (1840, 1850), (1850, 1860), (1860, 1870), (1870, 1880), (1880, 1890),
         (1890, 1900), (1900, 1910), (1910, 1920), (1920, 1930), (1930, 1940), (1940, 1950), (1950, 1960), (1960, 1970),
         (1970, 1980), (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2020), (2020, 2030), (2030, 2040), (2040, 2050),
         (2050, 2060), (2060, 2070), (2070, 2080), (2080, 2090), (2090, 2100), (2100, 2110), (2110, 2120), (2120, 2130),
         (2130, 2140), (2140, 2150), (2150, 2160), (2160, 2170), (2170, 2180), (2180, 2190), (2190, 2200), (2200, 2210),
         (2210, 2220), (2220, 2230), (2230, 2240), (2240, 2250), (2250, 2260), (2260, 2270), (2270, 2280), (2280, 2290),
         (2290, 2300), (2300, 2310), (2310, 2320), (2320, 2330), (2330, 2340), (2340, 2350), (2350, 2360), (2360, 2370),
         (2370, 2380), (2380, 2390), (2390, 2400), (2400, 2410), (2410, 2420), (2420, 2430), (2430, 2440), (2440, 2450),
         (2450, 2460), (2460, 2470), (2470, 2480), (2480, 2490), (2490, 2500), (2500, 2510), (2510, 2520), (2520, 2530),
         (2530, 2540), (2540, 2550), (2550, 2560), (2560, 2570), (2570, 2580), (2580, 2590), (2590, 2600), (2600, 2610),
         (2610, 2620), (2620, 2630), (2630, 2640), (2640, 2650), (2650, 2660), (2660, 2670), (2670, 2680), (2680, 2690),
         (2690, 2700), (2700, 2710), (2710, 2720), (2720, 2730), (2730, 2740), (2740, 2750), (2750, 2760), (2760, 2770),
         (2770, 2780), (2780, 2790), (2790, 2800), (2800, 2810), (2810, 2820), (2820, 2830), (2830, 2840), (2840, 2850),
         (2850, 2860), (2860, 2870), (2870, 2880), (2880, 2890), (2890, 2900), (2900, 2910), (2910, 2920), (2920, 2930),
         (2930, 2940), (2940, 2950), (2950, 2960), (2960, 2970), (2970, 2980), (2980, 2990), (2990, 3000), (3000, 3010),
         (3010, 3020), (3020, 3030), (3030, 3040), (3040, 3050), (3050, 3060), (3060, 3070), (3070, 3080), (3080, 3090),
         (3090, 3100), (3100, 3110), (3110, 3120), (3120, 3130), (3130, 3140), (3140, 3150), (3150, 3160), (3160, 3170),
         (3170, 3180), (3180, 3190), (3190, 3200), (3200, 3210), (3210, 3220), (3220, 3230), (3230, 3240), (3240, 3250),
         (3250, 3260), (3260, 3270), (3270, 3280), (3280, 3290), (3290, 3300), (3300, 3310), (3310, 3320), (3320, 3330),
         (3330, 3340), (3340, 3350), (3350, 3360), (3360, 3370), (3370, 3380), (3380, 3390), (3390, 3400), (3400, 3410),
         (3410, 3420), (3420, 3430), (3430, 3440)]
