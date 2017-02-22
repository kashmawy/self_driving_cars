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
from transform import TopDownTransform
from moviepy.editor import VideoFileClip
from datetime import datetime


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


def threshold(image, ksize=3):
    gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    # dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    hsv_binary = hsv_threshold(image, thresh=(100, 255))

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


objpoints = []
imgpoints = []

def full_pipeline(input_image):

    global objpoints
    global imgpoints

    if objpoints == [] and imgpoints == []:
        (objpoints, imgpoints) = get_points()

    # t1 = datetime.now()
    # print("Undistort")
    output_image = cal_undistort(input_image, objpoints, imgpoints)
    # t2 = datetime.now()
    # print("Undistorted", (t2 - t1).microseconds)

    src = np.float32([[585, 456],
                     [699, 456],
                     [1029, 667],
                     [290, 667]])

    dst = np.float32([[300, 70],
                     [1000, 70],
                     [1000, 600],
                     [300, 600]])

    # t3 = datetime.now()
    # print("Threshold")
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    threshold_image = threshold(output_image)
    # t4 = datetime.now()
    # print("Thresholded", (t4 - t3).microseconds)

    # t5 = datetime.now()
    # print("Transform")
    transformed_image = transform(threshold_image, src, dst, img_size)
    # t6 = datetime.now()
    # print("Transformed", (t6 - t5).microseconds)

    # plt.imshow(transformed_image)
    # plt.show()

    # histogram = get_histogram(transformed_image)

    # t7 = datetime.now()
    # print("Detect lane lines")
    (left_lane, right_lane, out_image) = full_detect_lane_lines(transformed_image)
    # t8 = datetime.now()
    # print("Detected lane lines", (t8 - t7).microseconds)

    left_fitx, ploty = left_lane.get_lane_fit(transformed_image.shape[0])
    right_fitx, ploty = right_lane.get_lane_fit(transformed_image.shape[0])

    plt.imshow(out_image)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='red')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

    # t9 = datetime.now()

    # print("Overlay")
    img = overlay(left_lane, right_lane, input_image, input_image.shape)
    # t10 = datetime.now()
    # print("Overlayed", (t10 - t9).microseconds)

    plt.imshow(img)
    plt.show()

    plt.imshow(out_image)
    # plt.show()

    return img


def convert_video():
    input_clip = VideoFileClip('./video/project_video.mp4')
    output_clip = input_clip.fl_image(full_pipeline)
    output_clip.write_videofile('./output/project_video.mp4', audio=False)


# convert_video()
result = full_pipeline(imread('./test_images/test1.jpg'))

# plt.imshow(result)
# plt.show()