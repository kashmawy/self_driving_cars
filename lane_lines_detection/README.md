# Algorithm Description

## The pipeline of the algorithm looks like the following:
1. Capture the image: Capture the image that the algorithm will operate on.
2. Grayscale: Convert the image into gray scale.
3. Gaussian filter: Apply Gaussian filter to the image to remove noise.
4. Region of interest: Crop only the part of the image where we imagine our lane lines will be.
5. Canny edge: Apply canny edge algorithm to detect edges.
6. Hough Transform: Hough Transform in order to detect lines.
7. Get the average slope: Get the average slope for lines with positive slope and negative slope which represents both lane lines.
8. Calculate the y-max and interpolate the lines for each side: Get the maximum y coordinate of both sides and use the maximum y coordinate in both lane lines to calculate the corresponding x value (using the average slope) and then interpolate the line by connecting the line between (x value, y minimum) and (x value, y maximum).


## Parts of the pipeline where this is more likely to fail and could use enhancement:
4. Region of interest
  If the road has wider lanes or if the camera which captures these images is moved, the region of interest can crop the wrong part of the image removing the lane lines by mistake.

  In order to make this better, it would be good instead to calculate the coordinates of these region of interest dynamically, by first scanning the image to see where we think the lane lines are at, then afterwards cropping everything that is outside this region. We can calculate where we think the lane lines are at, by going outwards in each direction horizontally from where we think we are in the image until we hit a line, this way we can calculate the x min and x max for the region of interest. As for vertical, we can include all the image.

8. Calculate y-max and interpolate lines for each side:
  If the road has a curve, the interpolate of the line would fail to represent the actual curve the line has and will represent it as a line segment instead between two points.

  In order to make this better, it would be good if the algorithm instead create the best fit between multiple points as opposed only two points, that way a curve be represented more realistically as opposed to as a line segment.
