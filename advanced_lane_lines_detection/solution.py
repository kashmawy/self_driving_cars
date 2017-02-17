import numpy as np
import cv2
import glob
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
# %matplotlib qt


def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera and cv2.undistort()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (retval, cameraMatrix, distCoeffs, rvecs, tvecs) = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    result = cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)
    return result


def get_points():
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

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

            # img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    return (objpoints, imgpoints)


def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgradient = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binaryoutput = np.zeros_like(absgradient)
    binaryoutput[(absgradient >= thresh[0]) & (absgradient <= thresh[1])] = 1
    return binaryoutput


def hsv_threshold(img, thresh=(170, 255)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    return s_binary


def threshold(image, ksize=3):
    gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    hsv_binary = hsv_threshold(image, thresh=(70, 255))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary


def transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_size = (gray.shape[1], gray.shape[0])
    return cv2.warpPerspective(img, M, gray_size, flags=cv2.INTER_LINEAR)


def get_histogram(img):
    import numpy as np
    histogram = np.sum(img[(int)(img.shape[0] / 2):, :], axis=0)
    return histogram


(objpoints, imgpoints) = get_points()
input_image = imread('./test_images/test1.jpg')
output_image = cal_undistort(input_image, objpoints, imgpoints)
# threshold_image = threshold(output_image)
threshold_image = pipeline(output_image)

src = np.float32([
    [300, 707],
    [580, 463],
    [750, 463],
    [1146, 707]
])

dst = np.float32([
    [209, 713],
    [203, 60],
    [1134, 37],
    [1191, 707]
])

transformed_image = transform(output_image, src, dst)
histogram = get_histogram(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY))

plt.subplot(2, 1, 1)
plt.title('Original')
plt.imshow(input_image)
plt.subplot(2, 1, 2)
plt.title('Output')
plt.imshow(output_image)
plt.show()


# plt.subplot(3, 2, 1)
# plt.title('Original')
# plt.imshow(input_image)
# plt.subplot(3, 2, 2)
# plt.title('Output')
# plt.imshow(output_image)
# plt.subplot(3, 2, 3)
# plt.title('Threshold')
# plt.imshow(threshold_image, cmap='gray')
# plt.subplot(3, 2, 4)
# plt.title('Bird Eye View')
# plt.imshow(transformed_image)
# plt.subplot(3, 2, 5)
# plt.title('Histogram')
# plt.plot(histogram)
# plt.show()
