# from cached_property import cached_property
import numpy as np

X_METER_PER_PIXEL = 3.7/700
Y_METER_PER_PIXEL = 30/720


class Lane:
    def __init__(self, xs, ys):
        self.xs = xs * X_METER_PER_PIXEL
        self.ys = ys * Y_METER_PER_PIXEL

    def fit(self):
        return np.polyfit(self.ys, self.xs, 2)

    def p(self):
        return np.poly1d(self.fit())

    # @cached_property
    def p1(self):
        """first derivative"""
        return np.polyder(self.p())

    # @cached_property
    def p2(self):
        """second derivative"""
        return np.polyder(self.p(), 2)

    def curvature(self, y):
        """returns the curvature of the of the lane in meters"""
        adjusted_y = y * Y_METER_PER_PIXEL
        return ((1 + (self.p1()(adjusted_y) ** 2)) ** 1.5) / np.absolute(self.p2()(adjusted_y))

    def get_lane_fit(self, ysize):
        ploty = np.linspace(0, ysize - 1, ysize)
        line_fit = self.fit()[0] * ploty ** 2 + self.fit()[1] * ploty + self.fit()[2]
        return line_fit, ploty