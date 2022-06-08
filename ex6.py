from ex5 import *
import utilities
from gtsam.utils import plot
from gtsam import symbol
import tqdm
from PoseGraphData import PoseGraph


# ----------------------------------------------------6.1---------------------------------------------------------------
def get_first_bundle_info(db):
    keyframe1, keyframe2 = 0, 5
    _, _, bundle_data = adjust_bundle(db, keyframe1, keyframe2)
    marginals = bundle_data.get_marginals()
    optimized_values = bundle_data.optimized_values
    plot.plot_trajectory(1, optimized_values, marginals=marginals, scale=1, title="Pose's Covariance")
    plt.show()

    # Apply marginalization and conditioning at the last key frame
    keys = gtsam.KeyVector()
    keys.append(symbol('c', keyframe1))
    keys.append(symbol('c', keyframe2))
    information_mat_first_second = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    cond_cov_mat = np.linalg.inv(information_mat_first_second)

    first_camera = optimized_values.atPose3(symbol('c', keyframe1))
    second_camera = optimized_values.atPose3(symbol('c', keyframe2))
    relative_pose = first_camera.between(second_camera)

    print("Relative covariance between the frame poses:\n", cond_cov_mat, "\n")
    print("Relative poses of last key frames:\n", relative_pose)


# ----------------------------------------------------6.2---------------------------------------------------------------


def main():
    db = ex4.build_data()
    # 6.1:
    # get_first_bundle_info(db)
    # 6.2:
    bundles = utilities.get_bundles()
    _, _, bundles = adjust_all_bundles(db, utilities.fives)
    pose_graph = PoseGraph(utilities.fives, bundles)


if __name__ == '__main__':
    main()
