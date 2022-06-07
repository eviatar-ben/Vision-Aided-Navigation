import numpy as np
import matplotlib.pyplot as plt
import ex4
import utilities
import exs_plots
from BundleData import BundleData
import gtsam
from gtsam import symbol
from gtsam.utils.plot import plot_trajectory, set_axes_equal


# todo maybe in ex4 every track is missing the last frame
# todo check weather the Rt that's got from ex3 are corresponding to the frame
#  (maybe there is a 1 shift between the frames and the Rts??)
# todo: should i multiply by K?
# todo: len of frames in track is len(track) - 1 : th last frame is needed to be appended

def get_gtsam_k_matrix():
    k = utilities.K
    fx, fy, skew, cx, cy, b = k[0, 0], k[1, 1], k[0, 1], k[0, 2], k[1, 2], -utilities.M2[0, 3]
    return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, b)


# ----------------------------------------------------5.1---------------------------------------------------------------
def get_gtsam_pose_and_and_stereo_matrix(first_frame_cam_to_world_ex_mat, frame, gtsam_calib_mat):
    """
    this function gets the C_first-1 matrix (which C_first-1 = C_first -> World)
    a current frame and the calibration matrix.
    and return the pose and the stereo of the current frame relative to the first frame (first frame of he bundle)
    """
    camera_relate_to_first_frame_trans = utilities.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                           frame.global_extrinsic_mat)
    cur_cam_pose = utilities.reverse_ext(camera_relate_to_first_frame_trans)
    gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)
    frame.gtsam_stereo_camera = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)

    return frame.gtsam_stereo_camera, gtsam_left_cam_pose, first_frame_cam_to_world_ex_mat


def triangulate_from_last_frame(first_frame, last_frame):
    """
    this function receive first and last frames and return the gtsam last frame stereo camera
     and the position of the last camera,  relative to the first frame.
    """

    # C_first =  World -> C_first

    # C_first-1 = C_first -> World
    first_frame_cam_to_world_ex_mat = utilities.reverse_ext(first_frame.global_extrinsic_mat)

    #   C_first -> World -> C_last =    C_first -> C_last
    camera_relate_to_first_frame_trans = utilities.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                           last_frame.global_extrinsic_mat)
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
    factors = []
    values = gtsam.Values()

    gtsam_calib_mat = get_gtsam_k_matrix()
    last_frame = frames[-1]
    first_frame = frames[0]

    left_locations, right_locations = utilities.get_track_frames_with_features(db, track)
    last_frame_feature_l = left_locations[-1]
    last_frame_feature_r = right_locations[-1]

    xl = last_frame_feature_l[0]
    yl = last_frame_feature_l[1]
    xr = last_frame_feature_r[0]
    gtsam_stereo_point2_for_triangulation = gtsam.StereoPoint2(xl, xr, yl)

    gtsam_last_frame_stereo_to_triangulate_from, gtsam_left_cam_pose, first_frame_cam_to_world_ex_mat = \
        triangulate_from_last_frame(first_frame, last_frame)
    gtsam_p3d = gtsam_last_frame_stereo_to_triangulate_from.backproject(gtsam_stereo_point2_for_triangulation)

    # Update values dictionary
    p3d_sym = symbol("q", 0)
    values.insert(p3d_sym, gtsam_p3d)

    left_projections, right_projections = [], []
    for i, frame in enumerate(frames):
        gtsam_frame, gtsam_left_cam_pose, _ = get_gtsam_pose_and_and_stereo_matrix(first_frame_cam_to_world_ex_mat,
                                                                                   frame, gtsam_calib_mat)
        # Create camera symbol and update values dictionary
        left_pose_sym = symbol("c", frame.frame_id)
        values.insert(left_pose_sym, gtsam_left_cam_pose)

        # Measurement values
        measure_xl, measure_xr, measure_y = left_locations[i][0], right_locations[i][0], left_locations[i][1]
        gtsam_measurement_pt2 = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)

        # Project p34 on frame
        gtsam_projected_stereo_point2 = gtsam_frame.project(gtsam_p3d)
        xl, xr, y = gtsam_projected_stereo_point2.uL(), gtsam_projected_stereo_point2.uR(), gtsam_projected_stereo_point2.v()

        # Factor creation
        projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                             symbol("c", frame.frame_id), symbol("q", 0), gtsam_calib_mat)

        factors.append(factor)

        left_projections.append([xl, y])
        right_projections.append([xr, y])

    total_proj_dist = utilities.get_total_projection_error(left_projections, right_projections, left_locations,
                                                           right_locations)

    projection_factor_errors = utilities.get_projection_factors_errors(factors, values)

    # plots:
    exs_plots.present_gtsam_re_projection_track_error(total_proj_dist, track)
    exs_plots.plot_factor_re_projection_error_graph(projection_factor_errors, track)
    exs_plots.plot_factor_as_func_of_re_projection_error_graph(projection_factor_errors, total_proj_dist, track)


def triangulate_from_last_frame_and_project_to_all_frames(db, track):
    frames_in_track = [frame for frame in track.frames_by_ids.values()]

    # define gtsam.StereoCamera for each frame in track:

    # triangulate the last frame's feature in the given track and re-project this 3D point to the track's frames:
    triangulate_and_project(db, track, frames_in_track)


# ----------------------------------------------------5.2---------------------------------------------------------------

def add_track_factors(db, graph, track, first_frame_ind, last_frame_ind, gtsam_frame_to_triangulate_from,
                      gtsam_calib_mat, initial_estimate, landmark_symbols):
    track_frames_inside_the_bundle = [frame for frame in track.frames_by_ids.values()
                                      if first_frame_ind <= frame.frame_id <= last_frame_ind]
    # Track's locations in frames_in_window
    left_locations, right_locations = utilities.get_track_frames_with_features(db, track,
                                                                               track_frames_inside_the_bundle)
    assert len(left_locations) == len(track_frames_inside_the_bundle)
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
    landmark_symbols.add(p3d_sym)
    initial_estimate.insert(p3d_sym, gtsam_p3d)

    for i, frame in enumerate(track_frames_inside_the_bundle):
        # Measurement values
        measure_xl, measure_xr, measure_y = left_locations[i][0], right_locations[i][0], left_locations[i][1]
        gtsam_measurement_pt2 = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)

        # Factor creation
        projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                             symbol('c', frame.frame_id), p3d_sym, gtsam_calib_mat)

        # Add factor to the graph
        graph.add(factor)


def adjust_bundle(db, keyframe1, keyframe2):
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    k = get_gtsam_k_matrix()
    cameras_symbols, landmark_symbols = set(), set()

    pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas([1e-3] * 3 + [1e-2] * 3)  # todo: nothing here is clear
    # pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(
    #     np.array([(3 * np.pi / 180) ** 2] * 3 + [1.0, 0.3, 1.0]))  # todo: check maor's covariances

    frames_in_bundle = [db.frames[frame_id] for frame_id in range(keyframe1, keyframe2 + 1)]
    first_frame = frames_in_bundle[0]
    first_frame_cam_to_world_ex_mat = utilities.reverse_ext(first_frame.global_extrinsic_mat)  # first cam -> world
    cur_cam_pose = None
    for frame_id, frame in zip(range(keyframe1, keyframe2 + 1), frames_in_bundle):

        assert frame_id == frame.frame_id
        left_pose_symbol = symbol("c", frame.frame_id)
        cameras_symbols.add(left_pose_symbol)
        # first frame
        if frame_id == keyframe1:
            first_pose = gtsam.Pose3()
            graph.add(gtsam.PriorFactorPose3(left_pose_symbol, first_pose, pose_uncertainty))

        # Compute transformation of : Rt(world - > cur cam) *Rt(first cam -> world) = Rt(first cam -> cur cam)
        camera_relate_to_first_frame_trans = utilities.compose_transformations(first_frame_cam_to_world_ex_mat,
                                                                               frame.global_extrinsic_mat)
        # Convert this transformation to: cur cam -> first cam
        cur_cam_pose = utilities.reverse_ext(camera_relate_to_first_frame_trans)
        initial_estimate.insert(left_pose_symbol, gtsam.Pose3(cur_cam_pose))

    gtsam_last_left_cam_pose = gtsam.Pose3(cur_cam_pose)

    # For each track create measurements factors
    # todo: check weather those are the desired tracks? shouldnt it be all tracks totally inside the bundle?
    # list(db.get_tracks_ids_in_frame(frames_in_bundle[1].frame_id))
    tracks_ids_in_frame = db.get_tracks_ids_in_frame(first_frame.frame_id)
    tracks_in_frame = [db.tracks[track_id] for track_id in tracks_ids_in_frame]
    # todo : pu kavor hacelev!!!!!!!!!!!!!!!!!1
    tracks_in_frame = [db.tracks[track_id] for track_id in tracks_ids_in_frame if
                       db.tracks[track_id].get_last_frame_id() >= keyframe2]
    for track in tracks_in_frame:
        # Create a gtsam object for the last frame for making the projection at the function "add_factors"
        # todo : can go out from the loop
        gtsam_last_cam = gtsam.StereoCamera(gtsam_last_left_cam_pose, k)
        add_track_factors(db, graph, track, keyframe1, keyframe2, gtsam_last_cam, k, initial_estimate,
                          landmark_symbols)  # Todo: as before

    optimized_estimation = optimize_graph(graph, initial_estimate)
    bundle_data = BundleData(keyframe1, keyframe2, cameras_symbols, landmark_symbols, graph, initial_estimate)

    bundle_data.set_optimized_values(optimized_estimation)

    return graph, initial_estimate, bundle_data


def optimize_graph(graph, initial_estimate):
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    return result


def accum_scene(values, global_trans=None, plot=False):
    if global_trans is None:
        global_trans = gtsam.Pose3()
    camera_loc = []
    points = []
    for key in values.keys():
        try:
            p = global_trans.transformFrom(values.atPoint3(key))
            points.append(p)
        except RuntimeError:  # the key is for a Pose3, not a Point3
            t = global_trans.compose(values.atPose3(key)).translation()
            camera_loc.append(t)

    if not plot:
        return np.array(camera_loc), np.array(points)
    ax = plt.figure().gca()
    camera_loc = np.array(camera_loc)
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, -1], c='c', s=3)
    ax.scatter(camera_loc[:, 0], camera_loc[:, -1], c='r', s=5)


# ----------------------------------------------------5.3---------------------------------------------------------------

def adjust_all_bundles(db, keyframes):
    import tqdm
    cameras = [gtsam.Pose3()]
    landmarks = []
    for keyframe1, keyframe2 in tqdm.tqdm(keyframes):
        try:
            _, _, bundle_data = adjust_bundle(db, keyframe1, keyframe2)

            cameras.append(bundle_data.get_optimized_cameras_p3d())
            landmarks.append(bundle_data.get_optimized_landmarks_p3d())
        except:
            print(f"problem with bundle: {keyframe1, keyframe2}")

    return np.array(cameras), landmarks


def bundle_adjustment(db):
    # bundle_adjustment:
    gtsam_cameras_rel_to_bundle, all_landmarks_rel_to_bundle = adjust_all_bundles(db, utilities.fives)

    # gtsam_cameras_rel_to_bundle, all_landmarks_rel_to_bundle , _= adjust_all_bundles(db, [(0, 5), (5, 10)])

    # convert relative landmarks and cameras poses to world coordinate:
    gtsam_cameras_rel_to_world = utilities.gtsam_left_cameras_relative_trans(gtsam_cameras_rel_to_bundle)
    landmarks_rel_to_world = utilities.compute_landmarks_in_relate_first_movie_camera(gtsam_cameras_rel_to_world,
                                                                                      all_landmarks_rel_to_bundle)
    # ground truth:
    ground_truth_keyframes = [i[0] for i in utilities.fives]
    ground_truth = np.array(utilities.get_ground_truth_transformations())[ground_truth_keyframes]
    cameras_gt_3d = utilities.left_cameras_trajectory(ground_truth)

    # initial estimation:
    initial_estimate_cameras_poses = utilities.get_initial_estimate_cameras_poses(db.frames.values())[
        ground_truth_keyframes]

    # plot:
    cameras_3d = utilities.gtsam_left_cameras_trajectory(gtsam_cameras_rel_to_world)
    exs_plots.plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth(cameras=cameras_3d,
                                                                                 landmarks=landmarks_rel_to_world,
                                                                                 cameras_gt=cameras_gt_3d,
                                                                                 initial_estimate_poses=
                                                                                 initial_estimate_cameras_poses)


def main():
    db = ex4.build_data()
    # 5.1
    # track = utilities.get_track_in_len(db, 20, False)
    # triangulate_from_last_frame_and_project_to_all_frames(db, track)
    # keyframe1, keyframe2 = 0, 10
    # graph, initial_estimate, bundle_data = adjust_bundle(db, keyframe1, keyframe2)
    # factor_error_before_optimization = np.log(graph.error(initial_estimate))  # log-likelihood
    # # ----3D Trajectory----:
    # plot_trajectory(fignum=0, values=initial_estimate)
    # # set_axes_equal(0)
    # plt.savefig(fr"plots/ex5/Trajectory3D/Trajectory3D({keyframe1, keyframe2}).png")
    # factor_error_after_optimization = np.log(graph.error(bundle_data.optimized_values))  # log-likelihood
    # # ----2D Trajectory----:
    # exs_plots.plot_left_cam_2d_trajectory(bundle_data)
    # accum_scene(bundle_data.optimized_values, plot=True)
    # plt.savefig(fr"plots/ex5/Trajectory2D/Trajectory2D({keyframe1, keyframe2}).png")
    # # 5.2
    # # ----Factor Error Diffs:----:
    # utilities.present_factor_error_differences(factor_error_after_optimization, factor_error_before_optimization)

    # 5.3
    bundle_adjustment(db)
    # adjust_bundle(db, 149, 155)

    print("Finished successfully")


if __name__ == '__main__':
    main()
