import numpy as np

from SEAL.SplineFunction import SplineFunction
from SEAL.lib import knot_averages


class SplineSpace(object):
    """
    The SplineSpace object functions mostly as a wrapper around the SplineFunction
    objects, providing auxiliary methods, mostly for visualization.
    """

    def __init__(self, p, t):
        """
        Initializes a spline space.
        :param p: Spline degree, integer
        :param t: knot vector of knots, ndarray
        """

        self.p = p
        self.t = t
        self.n = len(t) - p - 1

        # TODO: Handle this properly
        assert self.n > 0, "The knot vector has to be of at least length p + 2 = {p}".format(p=p + 2)

    def __call__(self, c):
        """
        Overrides the __call__ operator for the Spline Space object.
        Given a set of coefficients of length self.n, returns a corresponding
        SplineFunction object.
        :param c: Set of spline coefficients, in arbitrary dimension >= 1
        :return: SplineFunction object
        """

        return SplineFunction(self.p, self.t, c)

    def parameter_values(self, resolution=100):
        """
        Returns a list of parameter values, uniformly spaced in the range t[0], t[n+p+1]
        :param resolution: int, number of uniformly spaced values
        :return: numpy array of :resolution: number of uniformly spaced values in the range t[0], t[n+p+1].
        """

        return np.linspace(self.t[0], self.t[-1], resolution)

    @property
    def basis(self):
        """
        :return: Returns the n B-splines as callable SplineFunction objects 
        """

        basis = []
        for i in range(self.n):
            c = [i == j for j in range(self.n)]
            basis.append(SplineFunction(self.p, self.t, c))

        return basis

    def __str__(self):
        """
        Overrides the __str__ method
        :return: info-string
        """

        return """
        Spline Space:
            Dimension       = {n}
            Degree          = {p}
            Number of knots = {t}
        """.format(n=self.n, p=self.p, t=len(self.t))

    def vdsa(self, f):
        """
        Given a callable function f defined on the knot vector of S,
        finds the variation diminishing spline approximation (VDSA) to f
        in the spline space S.
        
        :param f: callable function defined on knot vector
        :return: the variation diminishing spline approximation to f
        """
        vdsa_coefficients = [f(tau) for tau in knot_averages(self.t, self.p)]
        return SplineFunction(self.p, self.t, vdsa_coefficients)
