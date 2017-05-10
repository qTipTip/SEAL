from SEAL import SplineFunction, SplineSpace

import numpy as np
import matplotlib.pyplot as plt

p = 2
tau = [0, 0, 0, 1, 2, 2, 2]
t = [0, 0, 0, 0.5, 1, 1.5, 2, 2, 2]
c = [1, -1, 1, -1]
S = SplineSpace(p, tau)

f = S(c)
f_refined = f.refine(t)

cp = f.control_polygon
cp_refined = f_refined.control_polygon

x = S.parameter_values()
y = f(x)
y_refined = f_refined(x)

plt.plot(x, y)
plt.plot(x, y_refined)
plt.plot(*zip(*cp))
plt.plot(*zip(*cp_refined))
plt.show()

