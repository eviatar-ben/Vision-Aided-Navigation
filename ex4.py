import pickle


class DataBase:
    def __init__(self):
        self.tracks = {}
        self.frames = {}

    def add_track(self, track):
        self.tracks[track.track_id] = track

    def add_frame(self, frame):
        self.frames[frame.frame_id] = frame

    def get_tracks_ids_in_frame(self, frame_id):
        return self.frames[frame_id].get_tracks_ids()

    def get_frames_ids_in_track(self, track_id):
        return self.frames[track_id].get_frames_ids_in_track()

    def get_feature_location(self, frame_id, track_id):
        return self.tracks[track_id].frames_id_to_kp(frame_id)

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
        self.frames_ids = {}
        self.frames_id_to_kp = {}  # mapping frames_id to key points relative to the ? first image ?

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
        self.tracks_ids = []  # which tracks going through this frame maybe dictionary is needed
        Frame.frame_id_counter += 1

    def add_track_ids(self, track_id):
        self.tracks_ids.append(track_id)

    def get_tracks_ids(self):
        return self.tracks_ids


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
