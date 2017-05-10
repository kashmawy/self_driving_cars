Lidar topview with 3D Bounding Boxes from Kitti dataset
=======================================================

[The project details can be found here](lidar_topview.ipynb)

1. The bin file from the Kitti dataset that represent the lidar data is extracted into an image that represents the top view.
2. The labels for that scene is loaded.
3. The coordinates of the labels is transformed from the camera coordinates to the lidar topview coordinates to match the same coordinates used in the first step.
4. The lidar top view is plotted with the labels from point 3 plotted as 3D bounding boxes.
