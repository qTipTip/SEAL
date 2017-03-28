import numpy as np

from lib import evaluate_non_zero_basis_splines, index

class RegularKnotVectorException(Exception):
    pass

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

        assert len(self.c) == len(self.t) - self.p - 1

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        start = t[0]
        end = t[-1]
        if (np.all(start == t[:self.p+1]) and np.all(end == t[-self.p-1:])):
            self._t = t
        else:
            raise RegularKnotVectorException("The first p+1 knots must be equal, and the last p+1 knots must be equal")

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

    def __call__(self, x):
        if isinstance(x, (list, set, np.ndarray)):
            y = np.zeros(len(x))
            for i in range(len(x)):
                y[i] = self._evaluate_spline(x[i])
            return y
        else:
            return self._evaluate_spline(x)

    def _evaluate_spline(self, x):
        """
        Evaluates the spline function at some parameter value x.
        Note that x has to lie in the range prescribed by the knot vector, self.t.
        :param x: parameter value
        :return: f(x)
        """

        mu = index(x, self.t)
        B = evaluate_non_zero_basis_splines(x, mu, self.t, self.p)
        C = self.c[mu - self.p:mu - self.p + len(B)]
        B = np.reshape(B, (len(B), 1))

        result = sum([c * b for c, b in zip(C, B)])
        return result
