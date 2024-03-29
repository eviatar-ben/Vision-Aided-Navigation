import pickle
import numpy as np


class DataBase:
    def __init__(self):
        self.tracks = {}  # {track_id : track}
        self.frames = {}  # {frame_id : frame}
        # last matches from l0 to l1:
        self.last_matches = None
        # change the architecture to list of dictionary each list i corresponding to frame i
        # in the list dict {next_match:track_id}

        self.track_by_match = {}  # {(next_match, frame_id) :track_id} : i.e, {(10, 0): 6}

    def add_track(self, track):
        self.tracks[track.track_id] = track

    def set_last_match(self, last_match, frame_id, track_id):
        self.track_by_match[(last_match, frame_id)] = track_id

    def get_track_id_by_match(self, next_match, frame_id):
        return self.track_by_match[(next_match, frame_id)]

    # todo: might be redundant to save all frames in the database
    def add_frame(self, frame):
        self.frames[frame.frame_id] = frame

    def get_frame(self, frame_id, track_id):
        return self.tracks[track_id].get_frame(frame_id)

    def get_tracks_ids_in_frame(self, frame_id):
        """function that returns all the TrackIds that appear on a given FrameId"""
        frame = self.frames[frame_id]
        return frame.tracks_to_features.keys()

    def get_tracks_ids_in_frames(self, first_frame_id, second_frame_id):
        result = set()
        for frame_id in range(first_frame_id, second_frame_id + 1):
            frame = self.frames[frame_id]
            result |= set(frame.tracks_to_features.keys())
        return result

    def get_frames_in_tracks(self, track_id):
        """"a function that returns all the FrameIds that are part of a given TrackId."""
        track = self.tracks[track_id]
        return track.frames_by_ids.keys()

    def get_feature_location(self, frame_id, track_id):
        """function that for a given (FrameId, TrackId) pair returns:
           Feature locations of track TrackId on both left and right images as a triplet (xl, xr, y)"""

        frame = self.get_frame(frame_id, track_id)
        feature = frame.get_feature(track_id)
        xl, xr, y = feature.xl, feature.xr, feature.y

        return xl, xr, y

    def serialize(self, path=r"ex4_pickles\DB.pickle"):
        pickle_out = open(path, "wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    @staticmethod
    def load_from_file(path=r"ex4_pickles\DB.pickle"):
        pickle_in = open(path, "rb")
        data_base = pickle.load(pickle_in)
        return data_base

    def present_statistics(self):
        from statistics import mean
        print(f"Total number of tracks is: {len(self.tracks)}")
        print(f"Number of frames is: {len(self.frames)}")
        tracks_lengths = [len(track) for track in self.tracks.values()]
        min_track_len = min(tracks_lengths)
        max_track_len = max(tracks_lengths)
        mean_track_len = mean(tracks_lengths)
        print(f"Max track len is : {max_track_len}")
        print(f"Min track len is : {min_track_len}")
        print(f"Mean track len is : {mean_track_len}")

    def set_inliers_per(self, inliers_pers):
        for inliers_per, frame in zip(inliers_pers, self.frames.values()):
            frame.inliers_per = inliers_per

    def update_cameras(self):
        pickle_in = open(r'../ex5_pickles/global_cameras.pickle', "rb")
        global_cameras = pickle.load(pickle_in)
        for frame, global_camera in enumerate(global_cameras):
            self.frames[frame].global_extrinsic_mat = global_camera


class Track:
    track_id_counter = 0

    def __init__(self):
        self.track_id = Track.track_id_counter
        self.frames_by_ids = {}  # {frame_id : frame}
        # self.kps_ids_match_in_track_path = []
        self.last_l1_r1_feature = None  # l1_point, r1_point, matched_feature, cur_frame.frame_id
        Track.track_id_counter += 1

    def add_frame(self, frame):
        self.frames_by_ids[frame.frame_id] = frame

    def get_frame(self, frame_id):
        return self.frames_by_ids[frame_id]

    def get_last_frame_id(self):
        max_id = max(self.frames_by_ids)
        return max_id

    def get_first_frame_id(self):
        max_id = min(self.frames_by_ids)
        return max_id

    def __len__(self):
        return len(self.frames_by_ids)

    def __str__(self):
        return

    # def add_kps_ids_match_to_track_path(self, kps_ids_in_path_track):
    #     self.kps_ids_match_in_track_path.append(kps_ids_in_path_track)


class Frame:
    frame_id_counter = 0

    def __init__(self):
        self.frame_id = Frame.frame_id_counter
        # which tracks going through this frame maybe dictionary is needed {frame: kp in frame_id frame}
        self.tracks_to_features = {}  # {track_id : feature}
        self.outgoing = 0
        self.inliers_per = None
        self.relative_extrinsic_mat = None  # relative to the last frame
        self.global_extrinsic_mat = None  # relative to the very first frame
        self.gtsam_stereo_camera = None

        Frame.frame_id_counter += 1

    def add_feature(self, track_id, feature):
        self.tracks_to_features[track_id] = feature

    def get_feature(self, track_id):
        return self.tracks_to_features[track_id]

    def set_extrinsic_mat(self, ext):
        self.relative_extrinsic_mat = ext  # for bundle adjustment


class Feature:
    # todo maybe add track_if to feature's fields
    def __init__(self, xl, xr, y, matched_feature):
        self.xl = xl
        self.xr = xr
        self.y = y

        self.matched_feature = matched_feature  # {idx_l0:idx_l1}:
