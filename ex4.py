from ex4_Objects import *
import ex3


def build_data():
    db = DataBase()

    for i in range(100):
        # kp, matches01 = get_kps_in_frame(i)
        db.add_track()