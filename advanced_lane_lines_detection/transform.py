import cv2
import numpy as np

default_src_points = np.float32([[585.714, 456.34],
                                 [699.041, 456.34],
                                 [1029.17, 667.617],
                                 [290.454, 667.617]])

default_dst_points  = np.float32([[300, 70],
                                  [1000, 70],
                                  [1000, 600],
                                  [300, 600]])

class TopDownTransform:
    def __init__(self, src_points=default_src_points, dst_points=default_dst_points):
        self.src_points = src_points
        self.dst_points = dst_points

    def transform_matrix(self):
        return cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def inverse_transform_matrix(self):
        return cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def transform_to_top_down(self, img):
        return cv2.warpPerspective(img, self.transform_matrix(), img.shape[1::-1])

    def transform_from_top_down(self, img, size_img):
        return cv2.warpPerspective(img, self.inverse_transform_matrix(), size_img.shape[1::-1])

