from SEAL.SplineSpace import SplineSpace
from SEAL.TensorProductSplineFunction import TensorProductSplineFunction


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
