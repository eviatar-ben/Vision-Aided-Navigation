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

def reverse_ext(ext):
    """
    this function gets an extrinsic matrix  world coord -> camera coord
    and return the "opposite" camera coord -> world coord
    """
    R = ext[:, :3]
    t = ext[:, 3]

    rev_R = R.T
    rev_t = -rev_R @ t
    return np.hstack((rev_R, rev_t.reshape(3, 1)))


def get_track_frames_with_features(db, track):
    """
    given database and track the function will return an array of images (frames)
    of the corresponding track both left and right frames,
    and both frames with the features loaded in the frames
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
