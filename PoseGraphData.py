import gtsam
from gtsam import symbol
import numpy as np


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
            np.array([(3 * np.pi / 180) ** 2] * 3 + [1.0, 0.3, 1.0]))
        factor = gtsam.PriorFactorPose3(first_left_cam_sym, gtsam_cur_global_pose, pose_uncertainty)
        self.graph.add(factor)

        self.initial_estimate.insert(first_left_cam_sym, gtsam_cur_global_pose)

        prev_sym = first_left_cam_sym

        # Create factor for each pose and add it to the graph
        for i in range(1, len(self.rel_poses)):
            cur_sym = symbol('c', self.key_frames[i][0])
            gtsam_cur_global_pose = gtsam_cur_global_pose.compose(self.rel_poses[i])
            self.global_pose.append(gtsam_cur_global_pose)

            # Create factor
            noise_model = gtsam.noiseModel.Gaussian.Covariance(self.cov[i - 1])
            factor = gtsam.BetweenFactorPose3(prev_sym, cur_sym, self.rel_poses[i], noise_model)
            self.graph.add(factor)

            # Add initial estimate
            self.initial_estimate.insert(cur_sym,  gtsam_cur_global_pose)

            prev_sym = cur_sym


    def optimize(self):
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        self.optimized_values = self.optimizer.optimize()

    def get_optimized_graph_error(self):
        return self.graph.error(self.optimized_values)

    def get_initial_graph_error(self):
        return self.graph.error(self.initial_estimate)
