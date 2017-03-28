import numpy as np


class SplineFunction(object):

    def __init__(self, degree, knots, coefficients):
        """
        Initialize a spline function
        :param degree: The spline degree p >= 0
        :param knots: A set of p + n + 1 increasing real values, where n > 0.
        :param coefficients: The set of n spline coefficients, where each coefficient is a d-tuple.
        """
        self.p = degree
        self.c = np.array(coefficients, dtype=np.float64)
        self.t = np.array(knots, dtype=np.float64)
        self.d = self._determine_coefficient_space(self.c)

    def _determine_coefficient_space(self, c):
        """
        Determines the coefficient space.
        If the dimension is 1, we also reshape the (n,) array into a (n, 1) array, for
        later computation.
        :param c: 
        :return: 
        """
        if len(c.shape) == 1:
            self.c = c.reshape(len(c), 1)
            return 1
        return c.shape[1]