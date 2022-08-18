from ex5 import *
import utilities
from PoseGraphData import PoseGraph
import pickle


def pickle_pose_graph(pose_graph):
    pickle_out = open(r"ex7_pickles/pose_graph.pickle", "wb")
    pose_graph.optimizer = None  # todo: check weather the optimizer is indeed redundant
    pickle.dump(pose_graph, pickle_out)
    pickle_out.close()


def get_pose_graph(get_loaded=True):
    if get_loaded:
        pickle_in = open(r"ex7_pickles/pose_graph.pickle", "rb")
        pose_graph = pickle.load(pickle_in)
        return pose_graph

    bundles = utilities.get_bundles()
    keyframes = utilities.perfect_fives
    pose_graph = PoseGraph(keyframes, bundles)
    pose_graph.optimize()
    pose_graph.all_loop_closures()
    pickle_pose_graph(pose_graph)
    return pose_graph


def main():
    # todo edit and sync the keyframes
    pose_graph = get_pose_graph()

    # plot a match result of a single successful consensus match:
    exs_plots.plot_single_successful_consensus_match(294, 420)

    # plot 5 versions of the pose graph along the process
    exs_plots.plot_five_pose_graph_versions(pose_graph)

    # Plot a graph of the absolute location error for the whole pose graph both with and without loop closures:
    exs_plots.plot_pose_graph_absolute_location_error_before_and_after_loop_closure()

    # Plot a graph of the location uncertainty size for the whole pose graph both with and without loop closures
    exs_plots.plot_covariance_uncertainty_before_and_after_loop_closure(pose_graph)


if __name__ == '__main__':
    main()
