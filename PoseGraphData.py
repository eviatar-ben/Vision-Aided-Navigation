import gtsam
from gtsam import symbol
import numpy as np

import ex7_Objects
import utilities


class PoseGraph:
    def __init__(self, key_frames, bundles):
        cov, rel_poses = PoseGraph.get_all_relative_covariance_and_poses(bundles)
        self.global_pose = []
        self.cov = cov
        self.rel_poses = rel_poses
        self.key_frames = key_frames
        self.optimizer = None
        self.initial_estimate = gtsam.Values()
        self.optimized_values = None
        self.graph = gtsam.NonlinearFactorGraph()
        self.build_factor_graph()
        # todo: check weather the cov is the proper relative cov
        self.shortest_path = ex7_Objects.shortest_path_generator()
        self.loops = []
        self.camera_symbols = []
        self.cace_initial_estimate_before_loop_closure = gtsam.Values()
        self.initial_estimate = gtsam.Values()
        self.build_factor_graph()

    @staticmethod
    def get_bundle_relative_covariance_and_poses(bundle):
        """
        Computes relative poses between key frame and their relative covariance matrix for one bundle
        :return: Relative pose and covariance matrix
        """
        # Compute bundle marginals
        first_key_frame, second_key_frame = bundle.keyframe1, bundle.keyframe2
        marginals = bundle.get_marginals()

        # Apply marginalization and conditioning to compute the covariance of the last key frame pose
        # in relate to first key frame
        keys = gtsam.KeyVector()
        keys.append(symbol('c', first_key_frame))
        keys.append(symbol('c', second_key_frame))
        information_mat_first_second = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
        cond_cov_mat = np.linalg.inv(information_mat_first_second)

        # Compute relative pose
        first_camera_pose = bundle.optimized_values.atPose3(symbol('c', first_key_frame))
        second_camera_pose = bundle.optimized_values.atPose3(symbol('c', second_key_frame))
        relative_pose = first_camera_pose.between(second_camera_pose)

        return relative_pose, cond_cov_mat

    @staticmethod
    def get_all_relative_covariance_and_poses(bundles):
        """
        this Method return the relative poses and covariance matrices lists o
        """
        import tqdm
        rel_poses_lst, cov_mat_lst = [], []

        for i in tqdm.tqdm(range(len(bundles))):
            relative_pose, cond_cov_mat = PoseGraph.get_bundle_relative_covariance_and_poses(bundles[i])

            cov_mat_lst.append(cond_cov_mat)
            rel_poses_lst.append(relative_pose)

        return cov_mat_lst, rel_poses_lst

    def build_factor_graph(self):
        # Create first camera symbol
        gtsam_cur_global_pose = gtsam.Pose3()
        first_left_cam_sym = symbol('c', self.key_frames[0][0])

        self.global_pose.append(gtsam_cur_global_pose)

        pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([(3 * np.pi / 180) ** 2] * 3 + [0.01, 0.001, 0.01]))
        factor = gtsam.PriorFactorPose3(first_left_cam_sym, gtsam_cur_global_pose, pose_uncertainty)
        self.graph.add(factor)

        self.initial_estimate.insert(first_left_cam_sym, gtsam_cur_global_pose)

        prev_sym = first_left_cam_sym

        # Create factor for each pose and add it to the graph
        for i in range(len(self.rel_poses) - 1):
            cur_sym = symbol('c', self.key_frames[i + 1][0])
            gtsam_cur_global_pose = gtsam_cur_global_pose.compose(self.rel_poses[i])
            self.global_pose.append(gtsam_cur_global_pose)

            # Create factor
            noise_model = gtsam.noiseModel.Gaussian.Covariance(self.cov[i])
            factor = gtsam.BetweenFactorPose3(prev_sym, cur_sym, self.rel_poses[i], noise_model)
            self.graph.add(factor)

            # Add initial estimate
            self.initial_estimate.insert(cur_sym, gtsam_cur_global_pose)

            prev_sym = cur_sym

    def optimize(self):
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        self.optimized_values = self.optimizer.optimize()

    def get_initial_estimate(self):
        return self.initial_estimate

    def get_optimized_graph_error(self):
        return self.graph.error(self.optimized_values)

    def get_initial_graph_error(self):
        return self.graph.error(self.initial_estimate)

    def get_marginals(self):
        return gtsam.Marginals(self.graph, self.optimized_values)

    def all_loop_closure(self, kfs_num=None):
        import tqdm
        if not kfs_num:
            kfs_num = len(self.key_frames)
        for i in tqdm.tqdm(range(kfs_num - 1)):
            self.single_loop_closure(i)

    def single_loop_closure(self, cur_kf):
        loop_constrain = self.get_consensus_frame_track(cur_kf)
        if loop_constrain:
            self.add_loop_factors(loop_constrain, cur_kf)
            self.optimize()
        return loop_constrain

    def add_loop_factors(self, loop_prev_frames_tracks_tuples, cur_frame):
        cur_frame_sym = symbol('c', cur_frame)
        cur_frame_movie_ind = self.key_frames[cur_frame][0]
        cur_frame_loop = []
        self.create_and_add_factors(cur_frame_loop, cur_frame_movie_ind, cur_frame_sym, loop_prev_frames_tracks_tuples)
        self.loops.append([cur_frame, cur_frame_loop])

    def create_and_add_factors(self, cur_frame_loop, cur_frame_movie_ind, cur_frame_sym,
                               loop_prev_frames_tracks_tuples):
        for prev_frame, tracks in loop_prev_frames_tracks_tuples:
            cur_frame_loop.append(prev_frame)

            prev_frame_movie_ind = self.key_frames[prev_frame][0]
            rel_pose, rel_last_cam_cov_mat = kf_bundle.compute_rel_pose_with_bundle(prev_frame_movie_ind,
                                                                                    cur_frame_movie_ind, tracks)
            prev_frame_sym = symbol('c', prev_frame)
            noise_model = gtsam.noiseModel.Gaussian.Covariance(rel_last_cam_cov_mat)
            factor = gtsam.BetweenFactorPose3(prev_frame_sym, cur_frame_sym, rel_pose, noise_model)
            self.graph.add(factor)

    def get_consensus_frame_track(self, cur_kf):
        mahalanobis_similarity = self.get_similiar_kf_bymahalanobis_distance(cur_kf)
        similar_kf_after_consensus_match = []

        if len(mahalanobis_similarity) > 0:
            similar_kf_after_consensus_match = self.heavy_operation(cur_kf, mahalanobis_similarity,
                                                                    similar_kf_after_consensus_match)

        return similar_kf_after_consensus_match

    def heavy_operation(self, cur_kf, mahalanobis_similarity):
        # todo check the [0] of both tuples
        cur_frame_ind = self.key_frames[cur_kf][0]
        mahalanobis_dist_cand_frame_ind = np.array(self.key_frames)[mahalanobis_similarity][:,
                                          0]
        similar_kf_after_consensus_match = \
            utilities.heavy_operation(mahalanobis_dist_cand_frame_ind,
                                      mahalanobis_similarity,
                                      cur_frame_ind,
                                      70)
        return similar_kf_after_consensus_match

    def get_similiar_kf_bymahalanobis_distance(self, cur_kf_ind):
        def get_mahalanobis_dist(delta, cov):
            r_squared = delta.T @ np.linalg.inv(cov) @ delta
            return r_squared ** 0.5

        similar_kf = []
        cur_kf_mat = self.initial_estimate.atPose3(symbol('c', cur_kf_ind))
        self.searche_closure_in_pre_kf(cur_kf_ind, cur_kf_mat, get_mahalanobis_dist, similar_kf)
        if len(similar_kf) > 0:
            similar_kf.sort(key=lambda x: x[0])
            similar_kf = np.array(similar_kf[:3]).astype(int)[:, 1]
        return similar_kf

    def searche_closure_in_pre_kf(self, cur_kf_ind, cur_kf_mat, get_mahalanobis_dist, similar_kf):
        for prev_kf_ind in range(cur_kf_ind):
            shortest_path = self.shortest_path.find_shortest_path(prev_kf_ind, cur_kf_ind)
            estimated_relative_cov = self.get_estimated_relative_cov(shortest_path)

            prev_kf_mat = self.initial_estimate.atPose3(symbol('c', prev_kf_ind))
            cams_delta = utilities.gtsam_cams_delta(prev_kf_mat, cur_kf_mat)

            mahalanobis_dist = get_mahalanobis_dist(cams_delta, estimated_relative_cov)
            if mahalanobis_dist < 70:
                similar_kf.append([mahalanobis_dist, prev_kf_ind])

    def get_estimated_relative_cov(self, shortest_path):
        estimated_relative_cov = np.zeros((6, 6))
        for i in range(1, len(shortest_path)):
            edge = self.shortest_path.get_edge_between_vertices(shortest_path[i - 1], shortest_path[i])
            estimated_relative_cov += edge.get_cov()
        return estimated_relative_cov
