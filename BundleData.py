import numpy as np
import gtsam
import pickle


class BundleData:

    def __init__(self, keyframe1, keyframe2, cameras_symbols, landmark_symbols,
                 factor_graph, initial_values, optimized_values=None):
        self.keyframe1 = keyframe1
        self.keyframe2 = keyframe2
        self.cameras_symbols = cameras_symbols
        self.landmark_symbols = landmark_symbols
        self.factor_graph = factor_graph
        self.initial_values = initial_values
        self.optimized_values = optimized_values
        self.marginals = None

    def set_optimized_values(self, optimized_values):
        self.optimized_values = optimized_values

    def get_optimized_cameras_p3d(self):
        from gtsam import symbol
        cam_pose = self.optimized_values.atPose3(symbol('c', self.keyframe2))
        return cam_pose

    # def get_optimized_cameras_p3d_version2(self):
    #     cameras_poses = []
    #     for camera_sym in self.cameras_symbols:
    #         cam_pose = self.optimized_values.atPose3(camera_sym)
    #         cameras_poses.append([cam_pose.x(), cam_pose.y(), cam_pose.z()])
    #
    #     return cameras_poses

    # def get_optimized_cameras_p3d(self):
    #     cameras_poses = []
    #     for camera_sym in self.cameras_symbols:
    #         cam_pose = self.optimized_values.atPose3(camera_sym)
    #         cameras_poses.append(np.asarray([cam_pose.x(), cam_pose.y(), cam_pose.z()]))
    #
    #     return np.asarray(cameras_poses)

    def get_optimized_landmarks_p3d(self):
        landmarks = []
        for landmark_sym in self.landmark_symbols:
            landmark = self.optimized_values.atPoint3(landmark_sym)
            landmarks.append(landmark)

        return np.asarray(landmarks)

    def get_marginals(self):
        self.marginals = gtsam.Marginals(self.factor_graph, self.optimized_values)
        return self.marginals

