# we wish to find the linear spline interpolant to
# the following data set:
from SEAL import parametrize, linear_spline_interpolation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# make sure the data_set is of type np.ndarray
data_set = np.array([(0, 3, 2), (5, 2, 1), (6, 7, 2), (4, 7, 9), (12, 5, 8), (19, -5, 3)])
parameters = parametrize(data_set, data_type='curve')
f = linear_spline_interpolation(parameters, data_values=data_set)

# evaluate and plot the spline over its parameter domain
t_values = np.linspace(parameters[0], parameters[-1], 100)
f_values = f(t_values)
fig = plt.figure()
axs = Axes3D(fig)
axs.plot(*zip(*f_values))
plt.show()
