import numpy as np
import cv2
import glob
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
from detect import detect_lane_lines, full_detect_lane_lines
# %matplotlib qt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont



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
    gradx = abs_sobel_thresh(image, orient='x', thresh=(30, 100))
    grady = abs_sobel_thresh(image, orient='y', thresh=(30, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(40, 100))
    # dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    hsv_binary = hsv_threshold(image, thresh=(130, 255))

    combined = np.zeros_like(mag_binary)

    combined[(hsv_binary == 1) | ((grady == 1) & (gradx == 1)) | ((mag_binary == 1) & (gradx == 1))] = 1

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


def transform(img, src, dst, img_size):
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, img_size)


def get_histogram(img):
    histogram = np.sum(img[(int)(img.shape[0] / 2):, :], axis=0)
    return histogram


def distance_from_center(left, right, center):
    X_METER_PER_PIXEL = 3.7 / 700
    Y_METER_PER_PIXEL = 30 / 720
    TO_METER = np.array([[X_METER_PER_PIXEL, 0],
                      [0, Y_METER_PER_PIXEL]])

    center_dot = np.dot(TO_METER, center)

    right_x = right.p()(center_dot[1])
    left_x = left.p()(center_dot[1])

    return ((right_x + left_x)/2 - center_dot[0])



def overlay_text(image, text, pos=(0, 0), color=(255, 255, 255)):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./fonts/liberation-sans.ttf", 64)
    draw.text(pos, text, color, font=font)
    image = np.asarray(image)

    return image


def overlay(left_lane, right_lane, img, shape):
    left_curvature = left_lane.curvature(shape[0])
    right_curvature = right_lane.curvature(shape[0])
    center_point = (shape[1] / 2, shape[0])
    center_distance = distance_from_center(left_lane, right_lane, center_point)

    print("Left curvature", left_curvature)
    print("Right curvature", right_curvature)
    print("Center distance", center_distance)

    left_overlay = "Left curvature: {0:.2f}m".format(left_curvature)
    img = overlay_text(img, left_overlay, pos=(10, 10))

    right_overlay = "Right curvature: {0:.2f}m".format(right_curvature)
    img = overlay_text(img, right_overlay, pos=(10, 90))

    center_overlay = "Distance from center: {0:.2f}m".format(center_distance)
    img = overlay_text(img, center_overlay, pos=(10, 180))

    return img


def full_pipeline(input_image):
    (objpoints, imgpoints) = get_points()
    output_image = cal_undistort(input_image, objpoints, imgpoints)
    # threshold_image = threshold(output_image)
    # threshold_image2 = pipeline(output_image)
    # threshold_image = threshold(output_image)

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


    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    threshold_image = threshold(output_image)
    transformed_image = transform(threshold_image, src, dst, img_size)

    plt.imshow(transformed_image)
    plt.show()

    # histogram = get_histogram(transformed_image)

    (left_lane, right_lane, out_image) = full_detect_lane_lines(transformed_image)

    ploty = np.linspace(0, transformed_image.shape[0]-1, transformed_image.shape[0])
    left_fitx = left_lane.fit()[0] * ploty ** 2 + left_lane.fit()[1] * ploty + left_lane.fit()[2]
    right_fitx = right_lane.fit()[0] * ploty ** 2 + right_lane.fit()[1] * ploty + right_lane.fit()[2]

    plt.imshow(out_image)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='red')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

    img = overlay(left_lane, right_lane, input_image, gray.shape)
    plt.imshow(img)
    plt.show()

    return out_image


result = full_pipeline(imread('./test_images/test1.jpg'))
# plt.imshow(result)
# plt.show()

# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 10))
# f.tight_layout()
#
# ax1.set_title("Sliding windows used in detection")
# ax1.imshow(result, cmap="gray")
#
# ax2.set_title("Fitted lines found in detection")
# ax2.imshow(result, cmap="gray");
#
# # Left lane
# # import pytest; pytest.set_trace()
# left_p = np.poly1d(lanes.left.pixels.fit())
# left_xp = np.linspace(0, 720, 100)
#
# # import pytest; pytest.set_trace()
# ax2.plot(left_p(left_xp), left_xp)
#
# # Right lane
# right_p = np.poly1d(lanes.right.pixels.fit())
# right_xp = np.linspace(0, 720, 100)
# ax2.plot(right_p(right_xp), right_xp)
#
# plt.show();


# plt.subplot(2, 2, 1)
# plt.title('Original')
# plt.imshow(input_image)
# plt.subplot(2, 2, 2)
# plt.title('Threshold')
# plt.imshow(threshold_image, cmap='gray')
# plt.subplot(2, 2, 3)
# plt.title('Transformed')
# plt.imshow(transformed_image)
# plt.subplot(2, 2, 4)
# plt.title('Result')
# plt.imshow(result)
# plt.subplot(3, 1, 3)
# plt.plot(histogram)
# plt.show()

# import pytest; pytest.set_trace()

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