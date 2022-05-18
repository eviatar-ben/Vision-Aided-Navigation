import gtsam
import ex4
import utilities


def get_gtsam_frames(frames):
    gtsam_frames = []
    for frame in frames:
        gtsam_frames.append()


def f():
    db = ex4.build_data()
    track = utilities.get_track_in_len(db, 10, False)
    frames = track.frames_by_ids.values()
    gtsam_frames = get_gtsam_frames(frames)


if __name__ == '__main__':
    # 5.1
    f()
