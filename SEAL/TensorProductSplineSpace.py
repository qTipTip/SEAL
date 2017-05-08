import numpy as np

from SEAL.SplineSpace import SplineSpace
from SEAL.TensorProductSplineFunction import TensorProductSplineFunction
from SEAL.lib import knot_averages


class TensorProductSplineSpace(object):
    """
    The TensorProductSplineSpace object functions as a wrapper around
    SplineSpace objects, providing auxiliary methods, mostly for
    visualization.
    """

    def __init__(self, p, t):
        """
        Initializes a tensor product spline space
        :param p: list/tuple, degrees in each direction
        :param t: list/tuple, knots in each direction, [knots_1, knots_2, ...]
        """

        self.r = len(p)
        self.p = p
        self.t = t
        self.n = [len(t_i) - p_i - 1 for t_i, p_i in zip(t, p)]
        self.S = [SplineSpace(p_i, t_i) for p_i, t_i in zip(p, t)]

    def __call__(self, c):
        """
        Overrides the _call__ operator for the TensorProductSplineSpace
        object. Given a matrix/tensor of coefficients, returns a corresponding
        TensorProductSplineFunction object.
        :param c: np.ndarray, matrix of spline coefficients.
        :return: TensorProductSplineFunction
        """

        return TensorProductSplineFunction(self.p, self.t, c)

    def parameter_values(self, resolution=100):
        """
        Returns an array of parameter values, uniformly spaced in the range t[0], t[n+p+1]
        for each 
        :param resolution: int, number of uniformly spaced values
        :return: numpy array of :resolution: number of uniformly spaced values in the range
        """
        x_values = np.linspace(self.t[0][0], self.t[0][-1], resolution)
        y_values = np.linspace(self.t[1][0], self.t[1][-1], resolution)

        return x_values, y_values

    def __str__(self):
        """
        Overrides the __str__ method to give some information about the spline space. 
        :return: info-string
        """
        return """
        TP Spline Space: 
            Dimension       = {nx}/{ny}
            Degree          = {px}/{py}
            Number of knots = {tx}/{ty}
        """.format(nx=self.n[0], ny=self.n[1], px=self.p[0], py=self.p[1], tx=len(self.t[0]), ty=len(self.t[1]))

    def vdsa(self, f, function_type='scalar'):
        """
        Given a callable function f defined on the knot rectangle of S,
        finds the variation diminishing spline approximation (VDSA) to f
        in the spline space S.

        :param f: callable function defined on knot rectangle
        :param function_type: string, whether f is scalar or parametric. 
        :return: the variation diminishing spline approximation to f
        """
        nx, ny = self.n
        if function_type == 'scalar':
            dim = 1
        elif function_type == 'parametric':
            dim = 3
        else:
            dim = 1
        vdsa_coefficients = np.zeros(shape=(nx, ny, dim))
        x_averages = knot_averages(self.t[0], self.p[0])
        y_averages = knot_averages(self.t[1], self.p[1])

        for i, x in enumerate(x_averages):
            for j, y in enumerate(y_averages):
                vdsa_coefficients[i, j, :] = f(x, y)
        return TensorProductSplineFunction(self.p, self.t, vdsa_coefficients)

    @property
    def basis(self):
        return [s.basis for s in self.S]
