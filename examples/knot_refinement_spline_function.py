from SEAL import SplineFunction, SplineSpace

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

p = 2
tau = [0, 0, 0, 1, 2, 2, 2]
t = [0, 0, 0, 0.5, 1, 1.5, 2, 2, 2]
c = [(1, 0, -1), (-1, 2, 2), (1, 5, -1), (-1, -1, 4)]
S = SplineSpace(p, tau)

f = S(c)
f_refined = f.refine(t)

cp = f.control_polygon
cp_refined = f_refined.control_polygon

x = S.parameter_values()
y = f(x)
y_refined = f_refined(x)

fig = plt.figure()
ax = Axes3D(fig)

ax.plot(*zip(*y))
ax.plot(*zip(*y_refined))
ax.plot(*zip(*cp))
ax.plot(*zip(*cp_refined))
plt.show()

