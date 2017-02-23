import numpy as np
import cv2
import glob
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
from lane import Lane
# %matplotlib qt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from moviepy.editor import VideoFileClip
from datetime import datetime
from transform import TopDownTransform


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


def color_threshold(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    yellow = cv2.inRange(hsv, (20, 100, 100), (50, 255, 255))

    sensitivity_1 = 68
    white = cv2.inRange(hsv, (0, 0, 255-sensitivity_1), (255, 20, 255))

    sensitivity_2 = 60
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    white_2 = cv2.inRange(hsl, (0, 255-sensitivity_2, 0), (255, 255, sensitivity_2))
    white_3 = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

    # s_binary = np.zeros_like(img)
    # return s_binary[(yellow == 1) | (white_2 == 1) | (white_3 == 1)]
    return yellow | white | white_2 | white_3


def threshold(image, ksize=3):
    # gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 100))
    # grady = abs_sobel_thresh(image, orient='y', thresh=(20, 100))
    # mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    # dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    # hsv_binary = hsv_threshold(image, thresh=(100, 255))
    color_binary = color_threshold(image)

    combined = np.zeros_like(color_binary)

    combined[(color_binary == 1)] = 1

    return color_binary


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

    center_dot = np.dot(center, TO_METER)

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


def overlay_lane(image, left_fit, right_fit, transform):
    left_ys = np.linspace(0, 100, num=101) * 7.2
    left_xs = left_fit[0]*left_ys**2 + left_fit[1]*left_ys + left_fit[2]

    right_ys = np.linspace(0, 100, num=101) * 7.2
    right_xs = right_fit[0]*right_ys**2 + right_fit[1]*right_ys + right_fit[2]

    color_warp = np.zeros_like(image).astype(np.uint8)

    pts_left = np.array([np.transpose(np.vstack([left_xs, left_ys]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_xs, right_ys])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, transform.inverse_transform_matrix(), (image.shape[1], image.shape[0]))
    newwarp = transform.transform_from_top_down(color_warp, image)

    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)


def overlay(left_lane, right_lane, img, shape):
    left_curvature = left_lane.curvature(shape[0])
    right_curvature = right_lane.curvature(shape[0])
    center_point = (shape[1] / 2, shape[0])
    center_distance = distance_from_center(left_lane, right_lane, center_point)

    # print("Left curvature", left_curvature)
    # print("Right curvature", right_curvature)
    # print("Center distance", center_distance)

    img = overlay_lane(img, left_lane.pixels_fit(), right_lane.pixels_fit(), TopDownTransform())

    left_overlay = "Left curvature: {0:.2f}m".format(left_curvature)
    img = overlay_text(img, left_overlay, pos=(10, 10))

    right_overlay = "Right curvature: {0:.2f}m".format(right_curvature)
    img = overlay_text(img, right_overlay, pos=(10, 90))

    center_overlay = "Distance from center: {0:.2f}m".format(center_distance)
    img = overlay_text(img, center_overlay, pos=(10, 180))

    return img


def detect_lane_lines(image, last_left_lane=None, last_right_lane=None):
    if last_left_lane is None or last_right_lane is None:
        return full_detect_lane_lines(image)

    return detect_lines_from_previous(image, last_left_lane, last_right_lane)

def detect_lines_from_previous(image, last_left_lane, last_right_lane):
    nonzero = image.nonzero()
    nonzero_x, nonzero_y = np.array(nonzero[1]), np.array(nonzero[0])

    last_left_p = np.poly1d(last_left_lane.pixels_fit())
    last_right_p = np.poly1d(last_right_lane.pixels_fit())

    margin = 100

    left_lane_indices = ((nonzero_x > (last_left_p(nonzero_y) - margin)) &
                         (nonzero_x < (last_left_p(nonzero_y) + margin)))


    right_lane_indices = ((nonzero_x > (last_right_p(nonzero_y) - margin)) &
                          (nonzero_x < (last_right_p(nonzero_y) + margin)))

    # Again, extract left and right line pixel positions
    left_x = nonzero_x[left_lane_indices]
    left_y = nonzero_y[left_lane_indices]
    right_x = nonzero_x[right_lane_indices]
    right_y = nonzero_y[right_lane_indices]

    if len(left_x) == 0 or len(left_y) == 0 or len(right_x) == 0 or len(right_y) == 0:
        return (last_left_lane, last_right_lane, image)


    return (Lane(left_x, left_y), Lane(right_x, right_y), image)


def full_detect_lane_lines(image):
    # Settings
    window_margin = 100          # This will be +/- on left and right sides of the window
    min_pixels_to_recenter = 50  # Minimum number of pixels before recentering the window
    num_windows = 9              # Number of sliding windows

    image_height, image_width = image.shape

    # Incoming image should already be  undistorted, transformed top-down, and
    # passed through thresholding. Takes histogram of lower half of the image.
    histogram = np.sum(image[image_height//2:,:], axis=0)

    # Placeholder for the image to be returned
    out_image = np.dstack((image, image, image))*255

    # Find peaks on left and right halves of the image
    midpoint = image_width//2
    base_left_x = np.argmax(histogram[:midpoint])
    base_right_x = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows based on num_windows
    window_height = image_height//num_windows

    # Get points of non-zero pixels in image
    nonzero = image.nonzero()
    nonzero_x, nonzero_y = np.array(nonzero[1]), np.array(nonzero[0])

    # Initialize current position, will be updated in each window
    current_left_x = base_left_x
    current_right_x = base_right_x

    # This is where the lane indices will be stored
    left_lane_indices = []
    right_lane_indices = []

    for window in range(num_windows):
        # Get the window boundaries
        window_y_low = image_height - (window + 1) * window_height
        window_y_high = image_height - window * window_height

        window_left_x_low = current_left_x - window_margin
        window_left_x_high = current_left_x + window_margin

        window_right_x_low = current_right_x - window_margin
        window_right_x_high = current_right_x + window_margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_image, (window_left_x_low, window_y_low), (window_left_x_high, window_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_image, (window_right_x_low, window_y_low), (window_right_x_high, window_y_high), (0, 255, 0), 2)

        # Identify the non-zero points within the window
        good_left_indices = ((nonzero_y >= window_y_low) &
                             (nonzero_y < window_y_high) &
                             (nonzero_x >= window_left_x_low) &
                             (nonzero_x < window_left_x_high)).nonzero()[0]

        good_right_indices = ((nonzero_y >= window_y_low) &
                              (nonzero_y < window_y_high) &
                              (nonzero_x >= window_right_x_low) &
                              (nonzero_x < window_right_x_high)).nonzero()[0]

        # Append the indices to the list
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)

        if(len(good_left_indices) > min_pixels_to_recenter):
            current_left_x = np.int(np.mean(nonzero_x[good_left_indices]))

        if(len(good_right_indices) > min_pixels_to_recenter):
            current_right_x = np.int(np.mean(nonzero_x[good_right_indices]))

    # Concatenate indices so it becomes a flat array
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract the land right lane pixels
    left_x = nonzero_x[left_lane_indices]
    left_y = nonzero_y[left_lane_indices]
    right_x = nonzero_x[right_lane_indices]
    right_y = nonzero_y[right_lane_indices]

    left_lane = Lane(left_x, left_y)
    right_lane = Lane(right_x, right_y)

    return (left_lane, right_lane, out_image)

objpoints = []
imgpoints = []
last_left_lane = None
last_right_lane = None

def full_pipeline(input_image):

    global objpoints
    global imgpoints
    global last_left_lane
    global last_right_lane

    if objpoints == [] and imgpoints == []:
        (objpoints, imgpoints) = get_points()

    output_image = cal_undistort(input_image, objpoints, imgpoints)

    src = np.float32([[585, 456],
                     [699, 456],
                     [1029, 667],
                     [290, 667]])

    dst = np.float32([[300, 70],
                     [1000, 70],
                     [1000, 600],
                     [300, 600]])

    gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    threshold_image = threshold(output_image)

    # plt.imshow(threshold_image, cmap='gray')
    # plt.show()

    transformed_image = transform(threshold_image, src, dst, img_size)

    # plt.imshow(transformed_image)
    # plt.show()

    (last_left_lane, last_right_lane, img) = detect_lane_lines(transformed_image, last_left_lane, last_right_lane)

    last_left_lane.get_lane_fit(transformed_image.shape[0])
    last_right_lane.get_lane_fit(transformed_image.shape[0])

    img = overlay(last_left_lane, last_right_lane, input_image, input_image.shape)

    return img


def convert_video(invideo, outvideo):
    input_clip = VideoFileClip(invideo)
    output_clip = input_clip.fl_image(full_pipeline)
    output_clip.write_videofile(outvideo, audio=False)


convert_video('./project_video.mp4', './output/output_video_6.mp4')
# result = full_pipeline(imread('./test_images/test1.jpg'))

# plt.imshow(result)
# plt.show()