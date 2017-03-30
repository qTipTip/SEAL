import matplotlib.pyplot as plt
import numpy as np

from SEAL.interpolation import cubic_hermite_interpolation

x_values = np.linspace(0, 2 * np.pi, 4)
f_values = np.sin(x_values)
df_values = np.cos(x_values)

plt.scatter(x_values, f_values)

data_set = np.column_stack((x_values, f_values, df_values))

x_values = np.linspace(0, 2 * np.pi, 100)
Hf = cubic_hermite_interpolation(data_set)
Hf_values = Hf(x_values)
plt.plot(x_values, np.sin(x_values))
plt.plot(x_values, Hf_values)
plt.show()
