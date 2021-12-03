import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('images/example_l.png', 0)
imgR = cv.imread('images/example_r.png', 0)
window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
stereo = cv.StereoSGBM_create(
    minDisparity=-1,
    numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=window_size,
    P1=8 * 3 * window_size,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size,
    disp12MaxDiff=12,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=32,
    preFilterCap=63,
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
