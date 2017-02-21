import numpy as np

X_METER_PER_PIXEL = 3.7/700
Y_METER_PER_PIXEL = 30/720

to_meters = np.array([[X_METER_PER_PIXEL, 0],
                      [0, Y_METER_PER_PIXEL]])

def in_meters(point):
    return np.dot(point, to_meters)

class Lanes:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def distance_from_center(self, center):
        center = in_meters(center)
        center_x, center_y = center

        right_x = self.right.meters.p(center_y)
        left_x = self.left.meters.p(center_y)

        return ((right_x + left_x)/2 - center_x)

    def lane_distance(self, y):
        _, y = in_meters((0, y))
        return (self.right.meters.p(y) - self.left.meters.p(y))

    def lanes_parallel(self, height, samples=50):
        distance_per_sample = height // samples
        distances = []
        for y in range(0, height, distance_per_sample):
            distances.append(self.lane_distance(y))

        std2 = 2*np.std(distances)
        mean = np.mean(distances)
        arr = np.array(distances)

        return len(arr[(arr > (mean + std2)) | (arr < (mean - std2))]) == 0
