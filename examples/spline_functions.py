import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from SplineFunction import SplineFunction
from lib import knot_averages

p = 2
t = [0, 0, 0, 1, 2, 3, 4, 4, 4]
n = len(t) - p - 1

c = [(1, -1, 3), (2, 1, 4), (4, 4, 5), (4, 9, 0), (5, 3, 1), (6, 6, -2)]
f = SplineFunction(p, t, c)
x_values = np.linspace(t[0], t[-1], 200)
y_values = f(x_values)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(*zip(*y_values))
ax.plot(*zip(*f.control_polygon()))
ax.scatter(*zip(*f.control_polygon()), s=50)
plt.show()

c = [-1, 1, -1, 1, -1, 1]
f = SplineFunction(p, t, c)
x_values = np.linspace(t[0], t[-1], 200)
y_values = f(x_values)
poly = f.control_polygon()

plt.plot(x_values, y_values)
plt.plot(*zip(*poly))
plt.show()