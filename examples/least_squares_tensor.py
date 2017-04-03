
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from SEAL.TensorProductSplineSpace import TensorProductSplineSpace
from SEAL.lib import create_knots


def f(x, y):
    return x, y, np.cos(np.sqrt(x**2 + y**2))

a = -10
b = 10
x_values = np.linspace(a, b, 20)
y_values = np.linspace(a, b, 20)

p = [3, 3]
n = [5, 5]

knots_x = create_knots(a, b, p[0], n[0])
knots_y = create_knots(a, b, p[1], n[1])

T = TensorProductSplineSpace(p, [knots_x, knots_y])
V = T.vdsa(f, function_type='parametric')
c = V.control_mesh
x, y = T.parameter_values(200)
v_vals = V(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(v_vals[:, :, 0], v_vals[:, :, 1], v_vals[:, :, 2], cmap='magma')
ax.plot_wireframe(c[:, :, 0], c[:, :, 1], c[:, :, 2])
ax.set_xlim3d(-10, 10)
ax.set_ylim3d(-10, 10)
ax.set_zlim3d(-10, 10)
plt.show()



