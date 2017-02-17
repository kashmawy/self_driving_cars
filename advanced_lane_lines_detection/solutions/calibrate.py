import numpy as np
import cv2
import glob
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
# %matplotlib qt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
# print("objp")
# print(objp)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# print("objp")
# print(objp)

def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera and cv2.undistort()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (retval, cameraMatrix, distCoeffs, rvecs, tvecs) = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    result = cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)
    return result

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        # cv2.imshow('img',img)
        # cv2.waitKey(500)


input_image = imread('./test_images/test1.jpg')
output_image = cal_undistort(input_image, objpoints, imgpoints)

plt.subplot(2, 1, 1)
plt.title('Original')
plt.imshow(input_image)
plt.subplot(2, 1, 2)
plt.title('Output')
plt.imshow(output_image)
plt.show()
