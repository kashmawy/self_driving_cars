# Note:
# The following code is heavily inspired from the code in the "Finding Lane
# Lines" lesson


# from lane import Lane
# from lanes import Lanes
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lane import Lane
from lanes import Lanes

def detect_lane_lines(image, last_left_lane=None, last_right_lane=None):
    if(last_lanes is None or last_right_lane is None):
        return full_detect_lane_lines(image)
    else:
        return quick_detect_lane_lines(image, last_left_lane, last_right_lane)

def quick_detect_lane_lines(image, last_left_lane, last_right_lane):
    nonzero = image.nonzero()
    nonzero_x, nonzero_y = np.array(nonzero[1]), np.array(nonzero[0])

    last_left_p = np.poly1d(last_left_lane.fit)
    last_right_p = np.poly1d(last_right_lane.fit)

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

    left = Lane(left_x, left_y)
    right = Lane(right_x, right_y)

    return (left, right, image)
    # return image

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

    out_image[nonzero_y[left_lane_indices], nonzero_x[left_lane_indices]] = [255, 0, 0]
    out_image[nonzero_y[right_lane_indices], nonzero_x[right_lane_indices]] = [0, 0, 255]



    ### The rest
    ###
    ###


    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((image, image, image)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzero_y[left_lane_indices], nonzero_x[left_lane_indices]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_indices], nonzero_x[right_lane_indices]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - window_margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + window_margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - window_margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + window_margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # plt.imshow(result)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    return out_image, ploty, left_fitx, right_fitx

    # left = Lane(left_x, left_y)
    # right = Lane(right_x, right_y)

    # return Lanes(left, right), out_image
    # return out_image

