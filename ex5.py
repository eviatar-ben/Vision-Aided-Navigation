import gtsam
from gtsam import symbol
import ex4
import utilities


# todo maybe in ex4 every track is missing the last frame
# todo check weather the Rt that's got from ex3 are corresponding to the frame
#  (maybe there is a 1 shift between the frames and the Rts??)
# todo: should i multiply by K?
# todo: len of frames in track is len(track) - 1 : th last frame is needed to be appended

def build_gtsam_frame(frame):
    reversed_ext = utilities.reverse_ext(frame.extrinsic_mat)
    R = reversed_ext[:, :3]
    t = reversed_ext[:, 3]
    gtsam_R = gtsam.Rot3(gtsam.Point3(R[:, 0]),
                         gtsam.Point3(R[:, 1]),
                         gtsam.Point3(R[:, 2]))
    gtsam_t = gtsam.Point3(t)
    gtsam_K = get_gtsam_k_matrix()

    left_pose = gtsam.Pose3(gtsam_R, gtsam_t)
    frame.gtsam_stereo_camera = gtsam.StereoCamera(left_pose, gtsam_K)
    return frame.gtsam_stereo_camera


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


def get_gtsam_k_matrix():
    k = utilities.K
    fx, fy, skew, cx, cy, b = k[0, 0], k[1, 1], k[0, 1], k[0, 2], k[1, 2], -utilities.M2[0, 3]
    return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, b)


#
# def get_factor(track_id, cam_id):
#     k = get_gtsam_k_matrix()  # todo: one matrix for all is enough?
#     projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)  # todo: nothing here is clear
#     a = gtsam.StereoPoint2(loc_on_cam[0][0], loc_on_cam[1][0], loc_on_cam[0][1])
#     return gtsam.GenericStereoFactor3D(a, projection_uncertainty, symbol('c', cam_id), symbol('q', track_id), k)

def add_track_factors(graph, track, first_frame_ind, last_frame_ind, gtsam_frame_to_triangulate_from, gtsam_calib_mat,
                      initial_estimate):
    # frames_in_track = [db.frames[frame.frame_id] for frame_id in db.frames[first_frame_ind: last_frame_ind + 1]
    frames_in_track = [db.frames[frame_id] for frame_id in range(first_frame_ind, last_frame_ind)]

    # Track's locations in frames_in_window
    left_locations, right_locations = utilities.get_track_frames_with_features(db, track)
    # left_locations = track.get_left_locations_in_specific_frames(first_frame_ind, last_frame_ind)
    # right_locations = track.get_right_locations_in_specific_frames(first_frame_ind, last_frame_ind)


    # Track's location at the Last frame for triangulations
    last_left_img_loc = left_locations[-1]
    last_right_img_loc = right_locations[-1]

    # Create Measures of last frame for the triangulation
    measure_xl, measure_xr, measure_y = last_left_img_loc[0], last_right_img_loc[0], last_left_img_loc[1]
    gtsam_stereo_point2_for_triangulation = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)

    # Triangulation from last frame
    gtsam_p3d = gtsam_frame_to_triangulate_from.backproject(gtsam_stereo_point2_for_triangulation)

    # Add landmark symbol to "values" dictionary
    p3d_sym = symbol('q', track.get_id())
    initial_estimate.insert(p3d_sym, gtsam_p3d)

    for i, frame in enumerate(frames_in_track):
        # Measurement values
        measure_xl, measure_xr, measure_y = left_locations[i][0], right_locations[i][0], left_locations[i][1]
        gtsam_measurement_pt2 = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)

        # Factor creation
        projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                             symbol('c', frame.frame_id), p3d_sym, gtsam_calib_mat)

        # Add factor to the graph
        graph.add(factor)


def adjust_bundle(db, keyframe1, keyframe2, computed_tracks, window_siz=10):
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    k = get_gtsam_k_matrix()

    pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas([1e-3] * 3 + [1e-2] * 3)  # todo: nothing here is clear
    # pose_uncertainty = np.array([(3 * np.pi / 180) ** 2] * 3 + [1.0, 0.3, 1.0])  # todo: check maor's covariances
    tracks_id_in_bundle = set()

    frames_in_bundle = [db.frames[frame_id] for frame_id in range(keyframe1, keyframe2)]
    first_frame = frames_in_bundle[0]

    first_frame_cam_to_world_ex_mat = utilities.reverse_ext(first_frame.extrinsic_mat)  # first cam -> world
    cur_cam_pose = None
    for frame_id, frame in zip(range(keyframe1, keyframe2), frames_in_bundle):
        left_pose_symbol = symbol('C', frame.frame_id)
        # first frame
        if frame_id == keyframe1:
            first_pose = gtsam.Pose3()
            graph.add(gtsam.PriorFactorPose3(left_pose_symbol, first_pose, pose_uncertainty))

        # Compute transformation of : Rt(world - > cur cam) *Rt(first cam -> world) = Rt(first cam -> cur cam)
        camera_relate_to_first_frame_trans = utilities.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                               frame.extrinsic_mat)
        # Convert this transformation to: cur cam -> first cam
        cur_cam_pose = utilities.reverse_ext(camera_relate_to_first_frame_trans)
        initial_estimate.insert(left_pose_symbol, cur_cam_pose)

    gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)

    # For each track create measurements factors
    # todo: check weather those are the desired tracks? shouldnt it be all tracks totally inide the bundle?
    tracks_ids_in_frame = db.get_tracks_ids_in_frame(first_frame.frame_id)
    tracks_in_frame = [db.tracks[track_id] for track_id in tracks_ids_in_frame if
                       db.tracks[track_id].get_last_frame_id() < keyframe2]
    # tracks_in_frame = [db.tracks[track_id] for track_id in tracks_ids_in_frame]

    for track in tracks_in_frame:
        # # Check that this track has bot been computed yet and that it's length is satisfied
        # if track.get_id() in self.__computed_tracks or track.get_last_frame_ind() < self.__second_key_frame:
        #     continue

        if track.track_id in computed_tracks:
            continue

        # Create a gtsam object for the last frame for making the projection at the function "add_factors"
        gtsam_last_cam = gtsam.StereoCamera(gtsam_left_cam_pose, k)
        add_track_factors(graph, track, keyframe1, keyframe2, gtsam_last_cam, k, initial_estimate)  # Todo: as before

        computed_tracks.append(track.track_id)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    return graph, result


# ----------------------------------------------------5.1---------------------------------------------------------------
def triangulate_and_project(db, track, frames, gtsam_frames):
    from exs_plots import present_gtsam_re_projection_track_error

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
    # todo: check weather the order is corresponding - its probbably revered in one of them
    present_gtsam_re_projection_track_error(projected, measured, track.track_id)
    return


def triangulate_from_last_frame_and_project_to_all_frames(db):
    track = utilities.get_track_in_len(db, 10, False)
    frames = track.frames_by_ids.values()

    # define gtsam.StereoCamera for each frame in track:
    gtsam_frames = get_gtsam_frames(frames)

    # triangulate the last frame's feature in the given track and re-project this 3D point to the track's frames:
    triangulate_and_project(db, track, frames, gtsam_frames)


if __name__ == '__main__':
    db = ex4.build_data()
    # 5.1
    triangulate_from_last_frame_and_project_to_all_frames(db)
    # 5.2
    graph, result = adjust_bundle(db, 0, 4, [])
