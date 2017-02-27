from vehicle_detection import find_cars
from features_extract import extract_features
from image_utils import (
    draw_labeled_bboxes,
    apply_threshold,
    add_heat,
    apply_boxes_with_heat_and_threshold,
)

from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import time
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import glob

# Load images

print("Loading Images")
vehicles = glob.glob('./vehicles/*/*.png')
nonvehicles = glob.glob('./non-vehicles/*/*.png')
print("Loaded Images")

## only for debugging
vehicles = vehicles[0:50]
nonvehicles = nonvehicles[0:50]


## Training

print("Extracting Features")
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [440, 656] # Min and max in y to search in slide_window()

vehicles_features = extract_features(
    vehicles,
    color_space=color_space,
    spatial_size=spatial_size,
    hist_bins=hist_bins,
    orient=orient,
    pix_per_cell=pix_per_cell,
    cell_per_block=cell_per_block,
    hog_channel=hog_channel,
    spatial_feat=spatial_feat,
    hist_feat=hist_feat,
    hog_feat=hog_feat
)

nonvehicles_features = extract_features(
    nonvehicles,
    color_space=color_space,
    spatial_size=spatial_size,
    hist_bins=hist_bins,
    orient=orient,
    pix_per_cell=pix_per_cell,
    cell_per_block=cell_per_block,
    hog_channel=hog_channel,
    spatial_feat=spatial_feat,
    hist_feat=hist_feat,
    hog_feat=hog_feat
)
print("Extracted Features")

# import pytest; pytest.set_trace()
X = np.vstack((vehicles_features, nonvehicles_features)).astype(np.float64)

print("Scaling")
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
print("Scaled")

y = np.hstack((np.ones(len(vehicles_features)), np.zeros(len(nonvehicles_features))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X,
    y,
    test_size=0.2,
    random_state=rand_state
)

print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
print("Training")
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
print("Trained")
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

## Detection

print("Detection")
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

ystart = 400
ystop = 656
scale = 1.5

image_path = 'test_images/test1.jpg'
img = imread(image_path)

bboxes = find_cars(
    img,
    ystart,
    ystop,
    scale,
    svc,
    X_scaler,
    orient,
    pix_per_cell,
    cell_per_block,
    spatial_size,
    hist_bins
)

print("Detected")

draw_img = apply_boxes_with_heat_and_threshold(img, bboxes)
plt.imshow(draw_img)
plt.show()


#
# image = mpimg.imread('bbox-example-image.jpg')
# draw_image = np.copy(image)
#
# # Uncomment the following line if you extracted training
# # data from .png images (scaled 0 to 1 by mpimg) and the
# # image you are searching is a .jpg (scaled 0 to 255)
# #image = image.astype(np.float32)/255
#
# windows = slide_window(image, x_start_stop=[None, 400], y_start_stop=y_start_stop,
#                     xy_window=(96, 96), xy_overlap=(0.5, 0.5))
#
# hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
#                         spatial_size=spatial_size, hist_bins=hist_bins,
#                         orient=orient, pix_per_cell=pix_per_cell,
#                         cell_per_block=cell_per_block,
#                         hog_channel=hog_channel, spatial_feat=spatial_feat,
#                         hist_feat=hist_feat, hog_feat=hog_feat)
#
# window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
#
# plt.imshow(window_img)


### TODO: Tweak these parameters and see how the results change.