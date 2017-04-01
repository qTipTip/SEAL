import matplotlib.pyplot as plt
import numpy as np

from SEAL.interpolation import cubic_hermite_interpolation


def test_function(t):
    return np.cos(t), np.sin(t), t


def test_dfunction(t):
    return -np.sin(t), np.cos(t), 1


t_values = np.linspace(0, 2 * np.pi, 5)
f_values = np.array([test_function(t) for t in t_values])
df_values = np.array([test_dfunction(t) for t in t_values])

hf = cubic_hermite_interpolation(parameter_values=t_values, function_values=f_values, derivatives=df_values)
x_values = np.linspace(0, 2 * np.pi, 100)
hf_values = hf(x_values)
f_values = np.array([test_function(t) for t in x_values])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(*zip(*f_values))
ax.plot(*zip(*hf_values))
ax.scatter(*zip(*hf.control_polygon))
ax.plot(*zip(*hf.control_polygon))
plt.show()
