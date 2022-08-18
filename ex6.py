from ex5 import *
import utilities
from gtsam.utils import plot
from gtsam import symbol
import tqdm
from PoseGraphData import PoseGraph


# ----------------------------------------------------6.1---------------------------------------------------------------
def get_first_bundle_info(db):
    keyframe1, keyframe2 = 0, 10
    _, _, bundle_data = adjust_bundle(db, keyframe1, keyframe2)
    marginals = bundle_data.get_marginals()
    optimized_values = bundle_data.optimized_values
    plot.plot_trajectory(1, optimized_values, marginals=marginals, scale=1, title="Pose's Covariance")
    plt.savefig(r'./plots/ex6/first_bundle_poses.png')
    plt.show()

    # Covariance (marginalization and conditioning):
    keys = gtsam.KeyVector()
    keys.append(symbol('c', keyframe1))
    keys.append(symbol('c', keyframe2))
    information_mat_first_second = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    cond_cov_mat = np.linalg.inv(information_mat_first_second)

    # Relative pose:
    first_camera = optimized_values.atPose3(symbol('c', keyframe1))
    second_camera = optimized_values.atPose3(symbol('c', keyframe2))
    relative_pose = first_camera.between(second_camera)

    print("Relative covariance between the frame poses:\n", cond_cov_mat, "\n")
    print("Relative poses of last key frames:\n", relative_pose)


# ----------------------------------------------------6.2---------------------------------------------------------------


def main():
    db = ex4.build_data()
    # 6.1:
    get_first_bundle_info(db)
    # 6.2:
    # bundles = utilities.get_bundles()
    _, _, bundles = adjust_all_bundles(db, utilities.perfect_fives)
    pose_graph = PoseGraph(utilities.perfect_fives, bundles)
    pose_graph.optimize()
    initial_estimate_poses = pose_graph.initial_estimate
    optimized_poses = pose_graph.optimized_values
    # plot initial estimations:
    # Plot initial estimate trajectory
    gtsam.utils.plot.plot_trajectory(0, initial_estimate_poses, title="Initial estimate trajectory", project_2d=True)
    plt.tight_layout()
    plt.savefig(r'./plots/ex6/initial_estimation_trajectory.png')
    plt.show()
    # Plot optimized trajectory
    gtsam.utils.plot.plot_trajectory(1, optimized_poses, title="optimized trajectory", project_2d=True)
    plt.tight_layout()
    plt.savefig(r'./plots/ex6/optimized_trajectory.png')
    plt.show()
    # Optimized trajectory with covariance
    marginals = pose_graph.get_marginals()
    plot.plot_trajectory(2, optimized_poses, marginals=marginals,
                         title="Optimized poses with covariance", project_2d=True)
    plt.tight_layout()
    plt.savefig(r'./plots/ex6/optimized_trajectory.png')
    plt.show()

    # Graph error before and after optimization
    print("Initial graph error: ", pose_graph.get_initial_graph_error())
    print("optimized graph error: ", pose_graph.get_optimized_graph_error())


if __name__ == '__main__':
    main()
