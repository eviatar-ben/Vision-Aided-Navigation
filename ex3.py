import cv2
import ex2
import sys
import numpy as np
import pandas as pd
import plotly.express as px

if __name__ == '__main__':
    # 3.1:
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    img0_cloud, img1_cloud = ex2.get_cloud(ex2.FIRST_IMAGE), ex2.get_cloud(ex2.SECOND_IMAGE)
    # 3.2:

