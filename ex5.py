import gtsam
from gtsam import symbol
import ex4
import utilities
import numpy as np
import exs_plots


# todo maybe in ex4 every track is missing the last frame
# todo check weather the Rt that's got from ex3 are corresponding to the frame
#  (maybe there is a 1 shift between the frames and the Rts??)
# todo: should i multiply by K?
# todo: len of frames in track is len(track) - 1 : th last frame is needed to be appended


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
    p3d_sym = symbol('q', track.track_id)
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

    first_frame_cam_to_world_ex_mat = utilities.reverse_ext(first_frame.relative_extrinsic_mat)  # first cam -> world
    cur_cam_pose = None
    for frame_id, frame in zip(range(keyframe1, keyframe2), frames_in_bundle):
        left_pose_symbol = symbol('C', frame.frame_id)
        # first frame
        if frame_id == keyframe1:
            first_pose = gtsam.Pose3()
            graph.add(gtsam.PriorFactorPose3(left_pose_symbol, first_pose, pose_uncertainty))

        # Compute transformation of : Rt(world - > cur cam) *Rt(first cam -> world) = Rt(first cam -> cur cam)
        camera_relate_to_first_frame_trans = utilities.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                               frame.relative_extrinsic_mat)
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

def triangulate_from_last_frame(first_frame, last_frame):
    """
    this function receive first and last frames and return the gtsam last frame stereo camera
     and the position of the last camera,  relative to the first frame.
    """

    # C_first =  World -> C_first
    # C_first-1 = C_first -> World
    first_frame_cam_to_world_ex_mat = utilities.reverse_ext(first_frame.general_extrinsic_mat)

    #   C_first -> World -> C_last =    C_first -> C_last
    camera_relate_to_first_frame_trans = utilities.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                           last_frame.general_extrinsic_mat)
    # C_last -> C_first
    cur_cam_pose = utilities.reverse_ext(camera_relate_to_first_frame_trans)
    gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)

    gtsam_K = get_gtsam_k_matrix()

    last_frame.gtsam_stereo_camera = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_K)
    return last_frame.gtsam_stereo_camera, gtsam_left_cam_pose, first_frame_cam_to_world_ex_mat


def project_3d_point_in_frame(gtsam_frame, p):
    """
    this function project a 3d point to a given gtsam frame.
    and return the projected point and the point's coord
    """
    gtsam_projected_2d_point = gtsam_frame.project(p)
    return gtsam_projected_2d_point, gtsam_projected_2d_point.uL(), gtsam_projected_2d_point.uR(), gtsam_projected_2d_point.v()


def triangulate_and_project(db, track, frames):
    from exs_plots import present_gtsam_re_projection_track_error
    import matplotlib.pyplot as plt

    gtsam_calib_mat = get_gtsam_k_matrix()
    last_frame = frames[-1]
    first_frame = frames[0]

    left_locations, right_locations = utilities.get_track_frames_with_features(db, track)
    last_frame_feature_l = left_locations[-1]
    xl = last_frame_feature_l[0]
    yl = last_frame_feature_l[1]
    last_frame_feature_r = right_locations[-1]
    xr = last_frame_feature_r[0]
    gtsam_stereo_point2_for_triangulation = gtsam.StereoPoint2(xl, xr, yl)

    gtsam_frame_to_triangulate_from, gtsam_left_cam_pose, first_frame_cam_to_world_ex_mat = \
        triangulate_from_last_frame(first_frame, last_frame)
    gtsam_p3d = gtsam_frame_to_triangulate_from.backproject(gtsam_stereo_point2_for_triangulation)

    left_projections, right_projections = [], []
    for i, frame in enumerate(frames):
        # Create camera symbol and update values dictionary
        left_pose_sym = symbol("c", frame.frame_id)

        camera_relate_to_first_frame_trans = utilities.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                               frame.general_extrinsic_mat)

        cur_cam_pose = utilities.reverse_ext(camera_relate_to_first_frame_trans)
        gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)

        # Measurement values
        measure_xl, measure_xr, measure_y = left_locations[i][0], right_locations[i][0], left_locations[i][1]
        gtsam_measurement_pt2 = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)

        gtsam_frame = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)

        # Project p34 on frame
        gtsam_projected_stereo_point2 = gtsam_frame.project(gtsam_p3d)
        xl, xr, y = gtsam_projected_stereo_point2.uL(), gtsam_projected_stereo_point2.uR(), gtsam_projected_stereo_point2.v()

        # Factor creation
        projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                             symbol("c", frame.frame_id), symbol("q", 0), gtsam_calib_mat)

        left_projections.append([xl, y])
        right_projections.append([xr, y])

    present_gtsam_re_projection_track_error(left_projections, right_projections, left_locations, right_locations,
                                            track)


def triangulate_from_last_frame_and_project_to_all_frames(db, track):
    frames_in_track = [frame for frame in track.frames_by_ids.values()]

    # define gtsam.StereoCamera for each frame in track:

    # triangulate the last frame's feature in the given track and re-project this 3D point to the track's frames:
    triangulate_and_project(db, track, frames_in_track)


# ---------------------------------------------------------repo---------------------------------------------------------
def triangulate_from_last_frame_and_project_to_all_frames_repo(db, track):
    import matplotlib.pyplot as plt

    frames_in_track = list(track.frames_by_ids.values())
    frame_to_triangulate_from = frames_in_track[-1]
    first_frame = frames_in_track[0]

    # Locations in frames_in_window
    left_locations, right_locations = utilities.get_track_frames_with_features(db, track)

    # Last frame locations for triangulations
    last_left_img_coor = left_locations[-1]
    last_right_img_coor = right_locations[-1]
    xl, xr, y = last_left_img_coor[0], last_right_img_coor[0], last_left_img_coor[1]
    gtsam_stereo_point2_for_triangulation = gtsam.StereoPoint2(xl, xr, y)

    # Triangulation
    first_frame_cam_to_world_ex_mat = utilities.reverse_ext(first_frame.relative_extrinsic_mat)  # first cam -> world

    camera_relate_to_first_frame_trans = utilities.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                           frame_to_triangulate_from.relative_extrinsic_mat)

    cur_cam_pose = utilities.reverse_ext(camera_relate_to_first_frame_trans)
    gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)

    gtsam_calib_mat = get_gtsam_k_matrix()
    gtsam_frame_to_triangulate_from = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)

    gtsam_p3d = gtsam_frame_to_triangulate_from.backproject(gtsam_stereo_point2_for_triangulation)

    left_projections = []
    right_projections = []

    for i, frame in enumerate(frames_in_track):
        # Create camera symbol and update values dictionary

        camera_relate_to_first_frame_trans = utilities.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                               frame.relative_extrinsic_mat)

        cur_cam_pose = utilities.reverse_ext(camera_relate_to_first_frame_trans)
        gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)

        # Measurement values
        measure_xl, measure_xr, measure_y = left_locations[i][0], right_locations[i][0], left_locations[i][1]
        gtsam_measurement_pt2 = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)

        gtsam_frame = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)

        # Project p34 on frame
        gtsam_projected_stereo_point2 = gtsam_frame.project(gtsam_p3d)
        xl, xr, y = gtsam_projected_stereo_point2.uL(), gtsam_projected_stereo_point2.uR(), gtsam_projected_stereo_point2.v()

        # Factor creation
        projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

        # Projections list
        left_projections.append([xl, y])
        right_projections.append([xr, y])

    # Compute projection euclidean error
    left_proj_dist = utilities.euclidean_dist(np.array(left_projections), np.array(left_locations))
    right_proj_dist = utilities.euclidean_dist(np.array(right_projections), np.array(right_locations))
    total_proj_dist = (left_proj_dist + right_proj_dist) / 2

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_title(f"Reprojection error for track: {track.track_id}")
    plt.scatter(range(len(total_proj_dist)), total_proj_dist)
    plt.ylabel('Error')
    plt.xlabel('Frames')
    fig.show()
    fig.savefig(fr"plots/ex5/gtsam_reprojection_error/gtsam_reprojection_track_error {track.track_id}.png")
    plt.close(fig)

    return total_proj_dist


if __name__ == '__main__':
    db = ex4.build_data()
    # 5.1
    track = utilities.get_track_in_len(db, 11, False)
    triangulate_from_last_frame_and_project_to_all_frames(db, track)
    # triangulate_from_last_frame_and_project_to_all_frames_repo(db, track)

    # 5.2
    # graph, result = adjust_bundle(db, 0, 4, [])
