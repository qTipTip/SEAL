import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use('fivethirtyeight')
import numpy as np

from SEAL.SplineSpace import SplineSpace
from SEAL.lib import create_knots

seal_data = np.loadtxt('../images/seal.dat', delimiter=',')[::-1]
n = len(seal_data)
p = 3
t = create_knots(0, 1, p, n)
S = SplineSpace(p, t)
f = S(seal_data)

x_values = np.linspace(0, 1, 1000)
f_values = f(x_values)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.grid('off')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.plot(*zip(*f_values), lw=3, zorder=-1)
ax.plot(*zip(*f.control_polygon), lw=1, alpha=0.3, c='green')
ax.scatter(*zip(*f.control_polygon), s=50, alpha=1, c='grey')

plt.tight_layout(w_pad=0, h_pad=0)
plt.gca().invert_yaxis()
plt.savefig('seal_spline.jpg')