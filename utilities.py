IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100


def get_track_in_len(db, track_min_len):
    # todo: check with random  track
    for track in db.tracks.values():
        if len(track) >= track_min_len:
            return track


def crop_image(xy, img, crop_size):
    """
    Crops image "img" to size of "crop_size" X "crop_size" around the coordinates "xy"
    :return: Cropped image
    """
    r_x = int(min(IMAGE_WIDTH, xy[0] + crop_size))
    l_x = int(max(0, xy[0] - crop_size))
    u_y = int(max(0, xy[1] - crop_size))
    d_y = int(min(IMAGE_HEIGHT, xy[1] + crop_size))

    return img[u_y: d_y, l_x: r_x], [crop_size, crop_size]
