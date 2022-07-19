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
        # todo: change implementation
        # todo: check weather the cov is the proper relative cov
        self.kf_graph = ex7_Objects.VertexGraph(len(key_frames), rel_covs=cov)

    # -----------------------------
    def create_vertex_graph(self):
        for i in range(len(self.cov)):
            self.add_edge(i, i + 1, self.cov[i])

    def add_edge(self, first_v, second_v, weight):
        self.kf_graph[first_v][second_v] = weight

    # -----------------------------

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

    def get_optimized_graph_error(self):
        return self.graph.error(self.optimized_values)

    def get_initial_graph_error(self):
        return self.graph.error(self.initial_estimate)

    def get_marginals(self):
        return gtsam.Marginals(self.graph, self.optimized_values)

    def all_loop_closure(self):
        import tqdm
        for i in tqdm.tqdm(range(len(self.key_frames))):
            self.loop_closure(i)

    def loop_closure(self, cur_kf):
        # mahalanobis_distance
        mahalanobis_similarity = self.mahalanobis_distance(cur_kf)
        similar_kf_after_consensus_match = []

        # if mahalanobis_similarity:
        #     frame_ind = np.array(self.key_frames)[cur_kf]  # frame index of the keyframe index
        #     similar_kf_after_consensus_match = find_loop_candidate_by_consensus_match(
        #         mahalanobis_dist_cand_at_movie_ind,
        #         mahalanobis_dist_cand_at_pg_ind,
        #         cur_frame_movie_ind,
        #         INLIERS_THRESHOLD_PERC)
        # return similar_kf_after_consensus_match


    def mahalanobis_distance(self, cur_kf_ind):
        # mahalanobis_dist = lambda delta, cov : (delta.T @ np.linalg.inv(cov) @ delta)** 0.5
        def mahalanobis_dist(delta, cov):
            r_squared = delta.T @ np.linalg.inv(cov) @ delta
            return r_squared ** 0.5

        similar_kf = []
        cur_kf_mat = self.initial_estimate.atPose3(symbol('c', cur_kf_ind))
        for prev_kf_ind in range(cur_kf_ind):
            # get shortest path:
            shortest_path = self.kf_graph.find_shortest_path(prev_kf_ind, cur_kf_ind)
            # get sum of relative covariance matrices between prev and cur kf:
            estimated_relative_cov = self.get_estimated_relative_cov(shortest_path)

            # get transformation between prev and cur kfs:
            prev_kf_mat = self.initial_estimate.atPose3(symbol('c', prev_kf_ind))

            cams_delta = utilities.gtsam_cams_delta(prev_kf_mat, cur_kf_mat)
            # get mahalanobis distance
            mahalanobis_dist = mahalanobis_dist(cams_delta, estimated_relative_cov)

            # check for threshold:
            if mahalanobis_dist < 100:
                similar_kf.append([mahalanobis_dist, prev_kf_ind])
        # sort, and take the 3 most similar
        if len(similar_kf) > 0:
            similar_kf.sort(key=lambda x: x[0])
            similar_kf = np.array(similar_kf[:3]).astype(int)[:, 1]
        return similar_kf

    def get_estimated_relative_cov(self, shortest_path):
        estimated_relative_cov = np.zeros((6, 6))
        for i in range(1, len(shortest_path)):
            edge = self.kf_graph.get_edge_between_vertices(shortest_path[i - 1], shortest_path[i])
            estimated_relative_cov += edge.get_cov()
        return estimated_relative_cov
