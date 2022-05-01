from ex4_Objects import *
import ex3
import pickle
import cv2
import numpy as np

FRAMES_NUM = 4


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
        track = Track()
        track.add_frame(first_frame)
        track.set_last_match(matched_feature)

        feature = Feature(l0_point[0], l0_point[1], first_frame.frame_id)
        first_frame.add_feature(track.track_id, feature)

        db.add_track(track)
        db.add_frame(first_frame)
    return first_frame


def extract_and_build_frame(db, l0_points, r0_points, supporters_matches01p, prev_frame, cur_frame):
    # prev_l1_match is the last match in the previous frame:
    for cur_l0, cur_l1 in supporters_matches01p.items():

        # the track is still going:
        if cur_l0 in db.last_matches.values():  # todo check sanity
            # extract the proper track:
            for track_id in prev_frame.tracks_to_features.keys():
                # if db.tracks[track_id].last_match == ():
                #     cur_track = prev_frame
                pass

        # new track
        else:
            track = Track()
            track.add_frame(cur_frame)
            pass
    print(supporters_matches01p)
    print(db.last_matches)
    a = 0


def build_data(data_pickled_already=True):
    tracks_data = get_tracks_data(data_pickled_already)
    db = DataBase()
    prev_frame = None
    first_frame = Frame()

    for track_data in tracks_data:
        first_frame_kp, second_frame_kp, supporters_matches01p = track_data[0], track_data[1], track_data[2]
        l0_points, r0_points = first_frame_kp
        l1_points, r1_points = second_frame_kp
        if not db.last_matches:
            prev_frame = extract_and_build_first_frame(db, l0_points, r0_points, supporters_matches01p, first_frame)
        else:
            cur_frame = Frame()
            extract_and_build_frame(db, l0_points, r0_points, supporters_matches01p, prev_frame, cur_frame)
            prev_frame = cur_frame


def main():
    build_data()
    # build_data(False)


if __name__ == '__main__':
    main()
