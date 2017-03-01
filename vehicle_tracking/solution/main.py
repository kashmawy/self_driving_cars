from vehicle_detection import (
    find_cars,
    slide_window,
    search_windows,
)
from features_extract import extract_features
from image_utils import (
    apply_threshold,
    add_heat,
    apply_boxes_with_heat_and_threshold,
    draw_boxes,
    convert_video,
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
import pickle
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--save-model', dest='save_model', type=str, help='save model')
parser.add_argument('--load-model', dest='load_model', type=str, help='load model')
parser.add_argument('--video', dest='video', type=str, help='the project video')
parser.add_argument('--output', dest='output', type=str, help='the output vide')

# Load images

global model
model = None

global X_scaler
X_scaler = None

global previous_boxes_list
previous_boxes_list = []

def load():
    print("Loading Images")
    vehicles = glob.glob('./vehicles/*/*.png')
    nonvehicles = glob.glob('./non-vehicles/*/*.png')
    print("Loaded Images")
    return (vehicles, nonvehicles)


def train(vehicles, nonvehicles):
    ## only for debugging
    # vehicles = vehicles[0:300]
    # nonvehicles = nonvehicles[0:300]


    ## Training

    print("Extracting Features")
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 4 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

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
    global X_scaler
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
    global model
    model = LinearSVC()
    # Check the training time for the SVC
    print("Training")
    t=time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    print("Trained")
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(model.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

    args = parser.parse_args()
    if args.save_model is not None:
        with open(args.save_model, 'wb') as fid:
            pickle.dump(model, fid)
        with open(args.save_model + '.2', 'wb') as fid:
            pickle.dump(X_scaler, fid)

    return model


def detect(image):
    scale = 1.5
    orient = 9  # HOG orientations
    pix_per_cell = 4 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    ystart = 400
    ystop = 656

    global X_scaler
    global model
    global previous_boxes_list

    bboxes = find_cars(
            image,
            ystart,
            ystop,
            scale,
            model,
            X_scaler,
            orient,
            pix_per_cell,
            cell_per_block,
            spatial_size,
            hist_bins,
            spatial_feat,
            hist_feat,
        )

    draw_img = apply_boxes_with_heat_and_threshold(image, bboxes, 13, previous_boxes_list)

    previous_boxes_list += [bboxes]
    if len(previous_boxes_list) == 16:
        del previous_boxes_list[0]

    return draw_img


args = parser.parse_args()
if args.load_model is not None:
    with open(args.load_model, 'rb') as fid:
        global model
        model = pickle.load(fid)
    with open(args.load_model + '.2', 'rb') as fid:
        global X_scaler
        X_scaler = pickle.load(fid)

else:
    (vehicles, nonvehicles) = load()
    model = train(vehicles, nonvehicles)

convert_video(args.video, args.output, detect)

# images = [
#     'test_images/test1.jpg',
#     'test_images/test2.jpg',
#     'test_images/test3.jpg',
#     'test_images/test4.jpg',
#     'test_images/test5.jpg',
#     'test_images/test6.jpg',
# ]
# for image_path in images:
#     img = imread(image_path)
#     (draw_img, heatmap, labels) = detect(img)
#     plt.subplot(3, 1, 1)
#     plt.imshow(draw_img)
#     plt.subplot(3, 1, 2)
#     plt.imshow(heatmap, cmap='hot')
#     plt.subplot(3, 1, 3)
#     plt.imshow(labels[0])
#     plt.plot()
#     plt.show()