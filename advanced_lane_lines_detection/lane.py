# from cached_property import cached_property
import numpy as np

X_METER_PER_PIXEL = 3.7/700
Y_METER_PER_PIXEL = 30/720

class PixelCalculations:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

        self.__fit = None
        self.__p = None
        self.__p1 = None
        self.__p2 = None

    # @cached_property
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
        return ((1 + (self.p1()*y**2))**1.5) / np.absolute(self.p2()*y)

class MeterCalculations(PixelCalculations):
    def __init__(self, xs, ys):
        PixelCalculations.__init__(self, xs * X_METER_PER_PIXEL, ys * Y_METER_PER_PIXEL)

    def curvature(self, y):
        """returns the curvature of the of the lane in meters"""
        return PixelCalculations.curvature(self, y * Y_METER_PER_PIXEL)


class Lane:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

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
        return ((1 + (self.p1()(y) ** 2)) ** 1.5) / np.absolute(self.p2()(y))