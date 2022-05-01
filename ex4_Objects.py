import pickle


class DataBase:
    def __init__(self):
        self.tracks = {}  # {track_id : track}
        self.frames = {}  # {frame_id : frame}

    def add_track(self, track):
        self.tracks[track.track_id] = track

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
        pickle_out = open(r"ex4_pickles\DB.pickle", "wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    @staticmethod
    def load_from_file(path=r"ex4_pickles\DB.pickle"):
        pickle_in = open(r"ex4_pickles\DB.pickle", "rb")
        data_base = pickle.load(pickle_in)
        return data_base


class Track:
    track_id_counter = 0

    def __init__(self):
        self.length = 0
        self.track_id = Track.track_id_counter
        self.frames_ids = {}  # {frame_id : frame}

        self.frames_id_to_kp = {}  # {frame_id: (kp_lo, kp_l1)} # mapping frames_id to key points

        self.kps_ids_match_in_track_path = []
        Track.track_id_counter += 1

    def add_frame(self, frame):
        self.frames_ids[frame.frame_id] = frame

    def add_kp_in_frame_id(self, frame, kps):
        self.frames_id_to_kp[frame.frame_id] = kps

    def get_frames_ids_in_track(self):
        return self.frames_ids

    def add_kps_ids_match_to_track_path(self, kps_ids_in_path_track):
        self.kps_ids_match_in_track_path.append(kps_ids_in_path_track)


class Frame:
    frame_id_counter = 0

    def __init__(self):
        self.frame_id = Frame.frame_id_counter
        # which tracks going through this frame maybe dictionary is needed {frame: kp in lo}
        self.tracks_ids = {}  # {track_id : track's_kp_in_frame}
        Frame.frame_id_counter += 1

    def add_track_ids(self, track):
        self.tracks_ids[track.track_id] = track

    def get_tracks_ids(self):
        return self.tracks_ids


def build_data_base():
    pass


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
