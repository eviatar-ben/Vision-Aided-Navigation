from ex5 import *
import utilities
from gtsam.utils import plot
from gtsam import symbol
import tqdm
from PoseGraphData import PoseGraph


def main():
    db = ex4.build_data()
    # 6.1:
    # 6.2:
    # bundles = utilities.get_bundles()
    # todo edit the keyframes
    keyframes = utilities.perfect_fives
    _, _, bundles = adjust_all_bundles(db, keyframes)
    pose_graph = PoseGraph(keyframes, bundles)
    pose_graph.optimize()
    pose_graph.all_loop_closure()


if __name__ == '__main__':
    main()
