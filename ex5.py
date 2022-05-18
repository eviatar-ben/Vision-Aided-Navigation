import gtsam
import ex4
import utilities


# todo maybe in ex4 every track is missing the last frame
# todo check wether the Rt thats got from ex3 are corresponding to the frame
#  (maybe there is a 1 shift between the frames and the Rts??)
# todo: should i multiply by K?


def build_gtsam_frame(frame):
    reversed_ext = utilities.reverse_ext(frame.extrinsic_mat)
    R = reversed_ext[:, :3]
    t = reversed_ext[:, 3]
    gtsam_R = gtsam.Rot3(gtsam.Point3(R[:, 0]),
                         gtsam.Point3(R[:, 1]),
                         gtsam.Point3(R[:, 2]))
    gtsam_t = gtsam.Point3(t)

    k = utilities.K
    fx, fy, skew, cx, cy = k[0, 0], k[1, 1], k[0, 1], k[0, 2], k[1, 2]
    gtsam_K = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -utilities.M2[0, 3])
    left_pose = gtsam.Pose3(gtsam_R, gtsam_t)
    gtsam_frame = gtsam.StereoCamera(left_pose, gtsam_K)
    return gtsam_frame


def get_gtsam_frames(frames):
    # gtsam_frames = [build_gtsam_frame(frame) for frame in frames]
    gtsam_frames = []

    for frame in frames:
        gtsam_frame = build_gtsam_frame(frame)
        gtsam_frames.append(gtsam_frame)

    # todo: maybe insert this gtsam as a frame's field
    return gtsam_frames


def triangulate(gtsam_frames):
    gtsam_last_frame_triangulated3D = gtsam_frames[-1].backproject()
    return


def f():
    db = ex4.build_data()
    track = utilities.get_track_in_len(db, 10, False)
    frames = track.frames_by_ids.values()
    # define gtsam.StereoCamera
    gtsam_frames = get_gtsam_frames(frames)
    errors = triangulate(gtsam_frames)


if __name__ == '__main__':
    # 5.1
    f()
