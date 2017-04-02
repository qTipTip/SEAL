import numpy as np

from SEAL.lib import index, evaluate_non_zero_basis_splines


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
        self.t = np.array(knots[0], dtype=np.float64), np.array(knots[1], dtype=np.float64)
        self.d = self._determine_coefficient_space(self.c, self.m)

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

        # single value
        if isinstance(x, (list, tuple, np.ndarray)) and isinstance(y, (list, tuple, np.ndarray)):
            f_values = np.zeros(shape=(len(x), len(y)))
            for i in range(len(x)):
                for j in range(len(y)):
                    f_values[i, j] = self._evaluate_spline(x[i], y[j])
            return f_values
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

        # TODO: Check this method. Something happens with the shape when
        # TODO: evaluated AT the endpoint!!!
        bx = evaluate_non_zero_basis_splines(x, mu_x, self.t[0], self.p[0])
        by = evaluate_non_zero_basis_splines(y, mu_y, self.t[1], self.p[1])
        coeff = self.c[mu_x - self.p[0]: mu_x + 1, mu_y - self.p[1]: mu_y + 1]

        # TODO: Do this properly, make sure that SplineFunction does not break,
        # TODO: if we change the evaluate_non_zero_basis_splines
        # Have to reshape, to get proper dimensions
        nx, ny = len(bx), len(by)
        bx = np.reshape(bx, (nx, 1))
        by = np.reshape(by, (ny, 1))
        # print(np.tensordot(bx.T.dot(coeff),by, 0))
        print('a', bx.T.shape)
        # print(coeff.dot(by))
        f = np.dot(bx.T.dot(np.squeeze(coeff)), by)
        return f
