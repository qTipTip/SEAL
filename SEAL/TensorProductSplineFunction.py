import numpy as np

from SEAL.SplineFunction import RegularKnotVectorException
from SEAL.lib import index, evaluate_non_zero_basis_splines, knot_averages


class TensorProductSplineFunction(object):
    def __init__(self, degree, knots, coefficients):
        """
        Initialize a tensor product spline function
        :param degree: Spline degrees in each direction 
        :param knots: A set of p_i + n_i + 1 increasing real values in each direction  
        :param coefficients: A matrix/tensor of dimension (m_1, m_2).
        """
        self.p = degree
        self.m = len(degree)
        self.c = np.array(coefficients, dtype=np.float64)
        self._t = None
        self.t = np.array(knots[0], dtype=np.float64), np.array(knots[1], dtype=np.float64)
        self.d = self._determine_coefficient_space(self.c, self.m)

    @property
    def t(self):
        return self._t

    # noinspection PyTypeChecker
    @t.setter
    def t(self, t):
        """
        Verifies that the knot vectors are p+1 regular.
        :param t: knot vectors
        :raises: RegularKnotVectorException
        """
        for i, knot_vector in enumerate(t):
            start = knot_vector[0]
            end = knot_vector[-1]
            if np.all(start == knot_vector[:self.p[i] + 1]) and np.all(end == knot_vector[-self.p[i] - 1:]):
                continue
            else:
                raise RegularKnotVectorException(
                    "The first p+1 knots must be equal, and the last p+1 knots must be equal")
        self._t = t

    def _determine_coefficient_space(self, c, m):
        """
        Determines the coefficient space.
        If the dimension is 1, we also reshape the (n_1, n_2) array
        into a (n_1, n_2, 1) array for later computation.
        :param c: coefficients
        :param m: number of axes
        :return: 
        """
        if len(c.shape) == m:
            self.c = self.c.reshape(tuple([m_i for m_i in c.shape] + [1]))
            return 1
        return c.shape[-1]

    def __call__(self, x, y):
        """
        Overrides the __call__ operator for the TensorProductSplineFunction.
        :param x: np.ndarray of shape (M,) or scalar value 
        :param y: np.ndarray of shape (N,) or scalar value
        :return: f(x, y) 
        """

        # array of values
        if isinstance(x, (list, tuple, np.ndarray)) and isinstance(y, (list, tuple, np.ndarray)):
            f_values = np.zeros(shape=(len(x), len(y), self.d))
            for i in range(len(x)):
                for j in range(len(y)):
                    f_values[i, j] = np.squeeze(self._evaluate_spline(x[i], y[j]))

            if self.d == 1:
                # scalar surface, then reshape to get rid of last axis of array
                return np.squeeze(f_values)
            else:
                # parametric surface, keep three axes.
                return f_values
        # scalar value
        else:
            return self._evaluate_spline(x, y)

    def _evaluate_spline(self, x, y):
        """
        Evaluates the TensorProductSplineFunction at some parameter value x = (x_1, x_2).
        Note that x has to lie in the range prescribed by the knot vector.
        :param x: parameter value
        :param y: parameter value
        :return: f(x, y)
        """

        mu_x = index(x, self.t[0])
        mu_y = index(y, self.t[1])

        bx = evaluate_non_zero_basis_splines(x, mu_x, self.t[0], self.p[0])
        by = evaluate_non_zero_basis_splines(y, mu_y, self.t[1], self.p[1])
        coeff = self.c[mu_x - self.p[0]: mu_x + 1, mu_y - self.p[1]: mu_y + 1]

        f = np.einsum('i,ijk,j->k', bx, coeff, by)  # compute the dot product over the first two axes.
        return f

    @property
    def control_mesh(self):

        if self.d == 1:
            knot_avg_x = knot_averages(self.t[0], self.p[0])
            knot_avg_y = knot_averages(self.t[1], self.p[1])
            control_mesh = np.zeros(shape=(len(knot_avg_x), len(knot_avg_x), 3))
            for i, x in enumerate(knot_avg_x):
                for j, y in enumerate(knot_avg_y):
                    control_mesh[i, j] = (x, y, self.c[i, j])
            return control_mesh
        else:
            return self.c
