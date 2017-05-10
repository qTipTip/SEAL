from SEAL import SplineSpace, create_knots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

p = 2
n = 10
t = create_knots(0, 1, p, n)
S = SplineSpace(p, t)

c = [(0, 1, 0), (1, 2, 1), (1.5, 3, 2), (1.7, -1, 3), (1, -1.5, 4), (3, 3, 3), (4, 4, 3), (5, 2, 2), (6, 5, 4), (7, -1, 5)]
f = S(c)
x = S.parameter_values()
y = f(x)

cp = f.visualize(iterations=4)
fig = plt.figure()
axs = Axes3D(fig)

axs.plot(*zip(*cp))
axs.plot(*zip(*y))
plt.show()