import numpy as np

from SEAL.lib import evaluate_non_zero_basis_splines, index, knot_averages, compute_fine_spline_coefficients, \
    insert_midpoints, evaluate_blossom


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
        self._t = None
        self.t = np.array(knots, dtype=np.float64)
        self.d = self._determine_coefficient_space(self.c)

        # TODO: Handle this better
        assert len(self.c) == len(self.t) - self.p - 1

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        """
        :param t: knot vector
        """
        self._t = t

    def _determine_coefficient_space(self, c):
        """
        Determines the coefficient space.
        If the dimension is 1, we also reshape the (n,) array into a (n, 1) array, for
        later computation.
        :param c: coefficients
        :return: 
        """
        if len(c.shape) == 1:
            self.c = c.reshape(len(c), 1)
            return 1
        return c.shape[1]

    def __call__(self, x):
        """
        Overrides the __call__ operator for the SplineFunction object.
        :param x: np.ndarray or float
        :return: f(x)
        """
        if isinstance(x, (list, set, np.ndarray)):
            y = np.ndarray(shape=(len(x), self.d))
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

        # TODO: Dot product here? More elegant
        result = sum([c * b for c, b in zip(C, B)])
        return result

    @property
    def control_polygon(self):
        """
        Returns the control polygon of the spline curve.
        Using knot averages if the curve is a spline function, and
        the spline coefficients if the curve is parametric.
        
        :return: a set of control points determining the control polygon 
        """

        if self.d == 1:
            knot_avg = knot_averages(self.t, self.p)
            return np.column_stack((knot_avg, self.c))
        else:
            return self.c

    def refine(self, refined_knots):
        """
        :param refined_knots: np.array representing a refinement of the knot vector self.t
        :return: SplineFunction, the same spline, represented in a finer space.
        """

        refined_coefficients = compute_fine_spline_coefficients(self.p, self.t, refined_knots, self.c)
        return SplineFunction(self.p, refined_knots, refined_coefficients)

    def visualize(self, iterations=5):
        """
        Returns the control polygon of the refined spline where midpoints
        have been inserted :iterations: number of times.
        :param iterations: Number of refinement
        :return: 
        """

        f = self
        t = self.t
        for i in range(iterations):
            t = insert_midpoints(t, self.p)
            f = f.refine(t)
        return f.control_polygon

    def evaluate_blossom(self, x, mu):
        """
        Returns the value of the p-variate polar form of the SplineFunction f restricted to the interval [t_mu, t_mu+1)..
        :param x: np.ndarray [x_1, ..., x_p]
        :return: B[f](x), the blossom of the spline function evaluated at x = [x_1, ..., x_p]
        """

        return evaluate_blossom(self.p, self.t, mu, self.c, x)
