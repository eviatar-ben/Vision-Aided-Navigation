from ex4_Objects import *
import ex3
import exs_plots
import utilities
import pickle
import cv2
import numpy as np

FRAMES_NUM = 3450


# todo: problem with the right stereo match. maybe the problem is different key points  slipping in

# todo: check whether last frame is not been considered?
def get_tracks_data(data_pickled_already):
    if not data_pickled_already:
        _, tracks_data = ex3.play(FRAMES_NUM)
        # tracks data consists kp - converts to xy points for the sake of uniformity
        tracks_data = convert_data_kp_to_xy_point(tracks_data)
    else:
        pickle_in = open(r"ex4_pickles\tracks_data.pickle", "rb")
        tracks_data = pickle.load(pickle_in)
    return tracks_data


def convert_data_kp_to_xy_point(tracks_data):
    converted_data_xy_points = []
    for data in tracks_data:
        first_frame_kp, second_frame_kp, supporters_matches01p = data[0], data[1], data[2]

        first_frame_p = cv2.KeyPoint_convert(np.asarray(first_frame_kp[0])), \
                        cv2.KeyPoint_convert(np.asarray(first_frame_kp[1]))
        second_frame_p = cv2.KeyPoint_convert(np.asarray(second_frame_kp[0])), cv2.KeyPoint_convert(
            np.asarray(second_frame_kp[0]))

        converted_data_xy_points.append([first_frame_p, second_frame_p, supporters_matches01p])
    return converted_data_xy_points


def extract_and_build_first_frame(db, l0_points, r0_points, supporters_matches01p, first_frame):
    db.last_matches = supporters_matches01p
    for l0_point, r0_point, matched_feature in zip(l0_points, r0_points, supporters_matches01p.items()):
        cur_l0, cur_l1 = matched_feature
        track = Track()
        track.add_frame(first_frame)
        db.set_last_match(cur_l1, first_frame.frame_id, track.track_id)

        feature = Feature(l0_point[0], r0_point[0], l0_point[1], matched_feature)
        first_frame.add_feature(track.track_id, feature)

        db.add_track(track)
    # in frame 0 all features are outgoing
    first_frame.outgoing = len(supporters_matches01p)
    db.add_frame(first_frame)
    return first_frame


def extract_and_build_frame(db, l0_points, r0_points, supporters_matches01p, prev_frame, cur_frame):
    # prev_l1_match is the last match in the previous frame:
    cur_frame_outgoing = 1  # consider the first frame
    for l0_point, r0_point, matched_feature in zip(l0_points, r0_points, supporters_matches01p.items()):
        cur_l0, cur_l1 = matched_feature

        # the track is still going:
        if cur_l0 in db.last_matches.values():  # todo check sanity
            cur_frame_outgoing += 1
            # extract the proper track:
            track_id = db.get_track_id_by_match(cur_l0, prev_frame.frame_id)
            track = db.tracks[track_id]
            track.add_frame(cur_frame)
            db.set_last_match(cur_l1, cur_frame.frame_id, track_id)
            # todo maybe add track_if to feature's fields
            feature = Feature(l0_point[0], r0_point[0], l0_point[1], matched_feature)
            cur_frame.add_feature(track.track_id, feature)
            # print(f"track: {track.track_id} is still going with length {len(track)}")
        # new track
        else:
            track = Track()
            track.add_frame(cur_frame)
            db.set_last_match(cur_l1, cur_frame.frame_id, track.track_id)

            feature = Feature(l0_point[0], r0_point[0], l0_point[1], matched_feature)
            cur_frame.add_feature(track.track_id, feature)

            db.add_track(track)
            # print(f"new track: {track.track_id} with length {len(track)}")
    cur_frame.outgoing = cur_frame_outgoing
    db.add_frame(cur_frame)

    db.last_matches = supporters_matches01p


def build_data(data_pickled_already=True):
    tracks_data = get_tracks_data(data_pickled_already)
    db = DataBase()
    prev_frame = None
    first_frame = Frame()
    inliers_pers = []

    for track_data in tracks_data:
        inliers_pers.append(track_data[3])
        first_frame_kps, second_frame_kps, supporters_matches01p = track_data[0], track_data[1], track_data[2]
        l0_points, r0_points = first_frame_kps
        l1_points, r1_points = second_frame_kps
        if not db.last_matches:
            prev_frame = extract_and_build_first_frame(db, l0_points, r0_points, supporters_matches01p, first_frame)
        else:
            cur_frame = Frame()
            extract_and_build_frame(db, l0_points, r0_points, supporters_matches01p, prev_frame, cur_frame)
            prev_frame = cur_frame

    db.set_inliers_per(inliers_pers)

    return db


# -----------------------------------------------------missions---------------------------------------------------------


def main():
    # 4.1
    db = build_data()
    # 4.2
    # db.present_statistics()
    # 4.3
    track = utilities.get_track_in_len(db, 10, True)
    exs_plots.display_track(db, track)
    # # 4.4
    # exs_plots.connectivity_graph(db.frames.values())
    # # 4.5
    # exs_plots.present_inliers_per_frame_percentage(db.frames.values())
    # # 4.6
    # exs_plots.present_track_len_histogram(db.tracks)
    # 4.7
    exs_plots.present_reprojection_error(db, track)


if __name__ == '__main__':
    # exs_plots.plot_ground_truth_2d()
    main()
