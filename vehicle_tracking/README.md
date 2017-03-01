**Vehicle Detection Project**

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this is in get_hog_features, which given an image and orient, pix_per_cell, cell_per_block and feature_vec extracts the HOG features in an image.

The following shows a vehicle, and the image transformed to YrCb, and the corresponding HOG for the first channel of YrCb:
![Vehicle with HOG](output_images/vehicle_with_hog.png)

The following shows a non vehicle with the corresponding HOG for the R Channel
![Non Vehicle with HOG](output_images/nonvehicle_with_hog.png)

These hog visualization are for images in YrCb format for only the first channel with the following setting:

1. orient of 9
2. pixels per cell of 4
3. cells per block of 2


####2. Explain how you settled on your final choice of HOG parameters.

I tried various parameters for HOG, and orient of 9, pixels per cell of 4 and cells per block of 2 seems to give the best results.

I tried using a much larger orient (e.g. 20), however that seemed to create so many bins for the different orientation and be more sensntive to differences in orientation even if they are close and not show matches.
I tried using bigger pixels per cell (e.g. 9), however that seemed to put a large area in one cell and the larger this number became, the harder it became to have a match because it must match a bigger portion now in order to match.
When pixels per cell is too small (e.g. 2) it causes a much smaller area to be in one cell and causes too many false positives because now matches doesn't really match enough cells to indicate that it is the same shape (vehicle or not)
Cells per block had the same behaviour.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this is in features_extract.py in the following methods extract_features, bin_spatial, color_hist, get_hog_features.
The code for training the SVM is in main.py.

I trained SVM using the following features:

1. Spatial binning (bin_spatial in features_extract.py)
2. Color Histogram (color_hist in features_extract.py)
3. HOG (get_hog_features in features_extract.py)

These are then concatenated by extract_features.
In main.py, all these are converted into a flat array, then afterwards fed into a Standard Scaler by removing the mean and scaling to unit variance.
Afterwards a support vector machine is trained using that data.
We have 8792 vehicle images and 8968 non vehicle images.
Off this data, 20% is reserved for testing and the rest 80% is used for training.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search code can be found in vehicle_detection.py in find_cars

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

