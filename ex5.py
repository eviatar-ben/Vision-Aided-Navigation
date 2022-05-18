import gtsam
import ex4
import utilities


def f():
    db = ex4.build_data()
    track = utilities.get_track_in_len(db, 11, False)


if __name__ == '__main__':
    #5.1
    f()