import numpy as np

from SEAL import SplineSpace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

p = 2
n = 5
t = np.linspace(0, 1, n + p + 1)
S = SplineSpace(p, t)

c = [(0, 1), (1, 3), (2, 1), (3, -1), (4, 2)]
f = S(c)
x = S.parameter_values()

cp = f.visualize(iterations=3)
fig = plt.figure()
axs = Axes3D(fig)

axs.plot(*zip(*cp))
axs.plot(*zip(*f.control_polygon))
plt.show()
