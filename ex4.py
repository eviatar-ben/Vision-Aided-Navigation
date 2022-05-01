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


def first_frame(db, l0_points, r0_points, supporters_matches01p):
    db.last_matches = supporters_matches01p
    frame = Frame()
    for l0_point, r0_point, matched_feature in zip(l0_points, r0_points, supporters_matches01p.items()):
        track = Track()
        track.add_frame(frame)

        feature = Feature(l0_point[0], l0_point[1], frame.frame_id)
        frame.add_track_ids(track)
        frame.add_feature_by_track_id(track.track_id, feature)

        db.add_track(track)
        db.add_frame(frame)


def extract_and_build_frame(db, l0_points, r0_points, supporters_matches01p):
    pass


def build_data(data_pickled_already=True):
    tracks_data = get_tracks_data(data_pickled_already)
    db = DataBase()

    for i, track_data in enumerate(tracks_data):
        i += 1  # consideing the first frame
        first_frame_kp, second_frame_kp, supporters_matches01p = track_data[0], track_data[1], track_data[2]
        l0_points, r0_points = first_frame_kp
        l1_points, r1_points = second_frame_kp

        if not db.last_matches:
            first_frame(db, l0_points, r0_points, supporters_matches01p)
        else:
            extract_and_build_frame(db, l0_points, r0_points, supporters_matches01p)


def main():
    build_data()
    # build_data(False)


if __name__ == '__main__':
    main()
