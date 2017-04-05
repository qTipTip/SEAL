import matplotlib.pyplot as plt
import numpy as np

from SEAL.SplineSpace import SplineSpace
from SEAL.approximation import least_squares_spline_approximation
from SEAL.lib import create_knots

n = 5
p = 2
a = 0
b = 10
t = create_knots(a=a, b=b, p=2, n=n)

S = SplineSpace(p, t)

x_values = np.linspace(a, b, 100)
y_values = np.sin(x_values), np.cos(x_values)
data_set = np.column_stack(y_values)
f = least_squares_spline_approximation(x_values, data_set, S)

x_values = np.linspace(a, b, 300)
f_values = f(x_values)
plt.plot(*zip(*f_values))
plt.show()
