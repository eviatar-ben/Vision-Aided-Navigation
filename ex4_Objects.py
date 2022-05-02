import pickle


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

    def get_feature_location(self, frame_id, track_id):
        frame = self.get_frame(frame_id, track_id)
        feature = frame.get_feature(track_id)
        xl, yl = feature.x, feature.y
        # xr, yr = kpl1.pt
        # assert yr == yl

        # return xl, xr, yl
        return xl, yl

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
        self.frames_by_ids = {}  # {frame_id : frame}
        # self.kps_ids_match_in_track_path = []
        Track.track_id_counter += 1

    def add_frame(self, frame):
        self.frames_by_ids[frame.frame_id] = frame

    def get_frame(self, frame_id):
        return self.frames_by_ids[frame_id]

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
        # which tracks going through this frame maybe dictionary is needed {frame: kp in lo}
        self.tracks_to_features = {}  # {track_id : feature}

        Frame.frame_id_counter += 1

    def add_feature(self, track_id, feature):
        self.tracks_to_features[track_id] = feature

    def get_feature(self, track_id):
        return self.tracks_to_features[track_id]


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
