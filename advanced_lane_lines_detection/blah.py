import matplotlib.pyplot as plt
import cv2
from scipy.misc import imread

input_image = imread('./test_images/test1.jpg')
cv2.rectangle(input_image, (10, 10), (100, 100), (0, 0, 0), 2)
plt.imshow(input_image)
plt.show()
