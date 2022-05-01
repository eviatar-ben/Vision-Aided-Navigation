import pickle


class DataBase:
    def __init__(self):
        self.tracks = {}  # {track_id : track}
        self.frames = {}  # {frame_id : frame}

        # last matches from l0 to l1:
        self.last_matches = None

    def add_track(self, track):
        self.tracks[track.track_id] = track

    # todo: might be redundant to save all frames in the database
    def add_frame(self, frame):
        self.frames[frame.frame_id] = frame

    def get_tracks_ids_in_frame(self, frame_id):
        return self.frames[frame_id].get_tracks_ids()

    def get_frames_ids_in_track(self, track_id):
        return self.frames[track_id].get_frames_ids_in_track()

    def get_kps_in_l0_r0(self, frame_id, track_id):
        return self.tracks[track_id].frames_id_to_kp(frame_id)

    def get_feature_location(self, frame_id, track_id):
        kpl0, kpl1 = self.get_kps_in_l0_r0(frame_id, track_id)
        xl, yl = kpl0.pt
        xr, yr = kpl1.pt
        assert yr == yl

        return xl, xr, yl

    def serialize(self, path=r"ex4_pickles\DB.pickle"):
        pickle_out = open(path, "wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    @staticmethod
    def load_from_file(path=r"ex4_pickles\DB.pickle"):
        pickle_in = open(path, "rb")
        data_base = pickle.load(pickle_in)
        return data_base


class Track:
    track_id_counter = 0

    def __init__(self):
        self.track_id = Track.track_id_counter
        self.frames_ids = {}  # {frame_id : frame}

        # self.kps_ids_match_in_track_path = []

        Track.track_id_counter += 1

    def add_frame(self, frame):
        self.frames_ids[frame.frame_id] = frame

    def __len__(self):
        return len(self.frames_ids)

    def __str__(self):
        return

    # def add_kps_ids_match_to_track_path(self, kps_ids_in_path_track):
    #     self.kps_ids_match_in_track_path.append(kps_ids_in_path_track)


class Frame:
    frame_id_counter = 0

    def __init__(self):
        self.frame_id = Frame.frame_id_counter
        # which tracks going through this frame maybe dictionary is needed {frame: kp in lo}
        self.tracks = {}  # {track_id : (x, y)}
        self.features = []

        self.track_id_to_kp = {}  # {track_id: (kp_lo, kp_l1)} # mapping frames_id to key points

        Frame.frame_id_counter += 1

    # todo: might be redundant to save all tracks in frame object
    def add_track_ids(self, track):
        self.tracks[track.track_id] = track

    def add_feature_by_track_id(self, track_id, feature):
        self.track_id_to_kp[track_id] = feature


class Feature:
    def __init__(self, x, y, matched_feature):
        self.x = x
        self.y = y
        self.matched_feature = matched_feature  # {idx_l0:idx_l1}:


if __name__ == '__main__':
    # db = DataBase()
    # t = Track()
    # db.add_track(t)
    # t2 = Track()
    # db.add_track(t2)
    # db.serialize()
    # unpickled = db.load_from_file()
    # print(unpickled.tracks)
    pass
