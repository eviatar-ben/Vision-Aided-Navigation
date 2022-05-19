import gtsam
import ex4
import utilities


# todo maybe in ex4 every track is missing the last frame
# todo check weather the Rt that's got from ex3 are corresponding to the frame
#  (maybe there is a 1 shift between the frames and the Rts??)
# todo: should i multiply by K?


def build_gtsam_frame(frame):
    reversed_ext = utilities.reverse_ext(frame.extrinsic_mat)
    R = reversed_ext[:, :3]
    t = reversed_ext[:, 3]
    gtsam_R = gtsam.Rot3(gtsam.Point3(R[:, 0]),
                         gtsam.Point3(R[:, 1]),
                         gtsam.Point3(R[:, 2]))
    gtsam_t = gtsam.Point3(t)

    k = utilities.K
    fx, fy, skew, cx, cy = k[0, 0], k[1, 1], k[0, 1], k[0, 2], k[1, 2]
    gtsam_K = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -utilities.M2[0, 3])
    left_pose = gtsam.Pose3(gtsam_R, gtsam_t)
    gtsam_frame = gtsam.StereoCamera(left_pose, gtsam_K)
    return gtsam_frame


def project_3d_point_in_frame(gtsam_frame, p):
    """
    this function project a 3d point to a given gtsam frame.
    and return the projected point and the point's coord
    """
    gtsam_projected_2d_point = gtsam_frame.project(p)
    return gtsam_projected_2d_point, gtsam_projected_2d_point.uL(), gtsam_projected_2d_point.uR(), gtsam_projected_2d_point.v()


def get_gtsam_frames(frames):
    # gtsam_frames = [build_gtsam_frame(frame) for frame in frames]
    gtsam_frames = []

    for frame in frames:
        gtsam_frame = build_gtsam_frame(frame)
        gtsam_frames.append(gtsam_frame)

    # todo: maybe insert this gtsam as a frame's field
    return gtsam_frames


def triangulate_and_project(db, track, frames, gtsam_frames):
    from  exs_plots import present_gtsam_re_projection_track_error

    features_in_frames_l_xy, features_in_frames_r_xy = utilities.get_track_frames_with_features(db, track)
    last_frame_feature_l = features_in_frames_l_xy[-1]
    xl = last_frame_feature_l[0]
    yl = last_frame_feature_l[1]
    last_frame_feature_r = features_in_frames_r_xy[-1]
    xr = last_frame_feature_r[0]
    gtsam_last_point2D = gtsam.StereoPoint2(xl, xr, yl)
    gtsam_last_frame_triangulated3D = gtsam_frames[-1].backproject(gtsam_last_point2D)

    projected_features_in_frames_l_xy, projected_features_in_frames_r_xy = [], []
    for gtsam_frame in gtsam_frames:
        _, xl, xr, y = project_3d_point_in_frame(gtsam_frame, gtsam_last_frame_triangulated3D)
        projected_features_in_frames_l_xy.append([xl, y])
        projected_features_in_frames_r_xy.append([xr, y])

    projected = [projected_features_in_frames_l_xy,
                 projected_features_in_frames_r_xy]
    measured = [features_in_frames_l_xy,
                features_in_frames_r_xy]
    #todo: check weather the order is corresponding - its probbably revered in one of them
    present_gtsam_re_projection_track_error(projected, measured, track.track_id)
    return


def triangulate_from_last_frame_and_project_to_all_frames():
    db = ex4.build_data()
    track = utilities.get_track_in_len(db, 10, True)
    frames = track.frames_by_ids.values()

    # define gtsam.StereoCamera for each frame in track:
    gtsam_frames = get_gtsam_frames(frames)

    # triangulate the last frame's feature in the given track and re-project this 3D point to the track's frames:
    triangulate_and_project(db, track, frames, gtsam_frames)


if __name__ == '__main__':
    # 5.1
    triangulate_from_last_frame_and_project_to_all_frames()
