import matplotlib.pyplot as plt
import numpy as np

from SEAL.interpolation import cubic_hermite_interpolation

x_values = np.linspace(0, 2 * np.pi, 6)
f_values = np.sin(x_values)
df_values = np.cos(x_values)

plt.scatter(x_values, f_values)

Hf_df = cubic_hermite_interpolation(x_values, f_values, df_values)
Hf_no_df = cubic_hermite_interpolation(x_values, f_values)

x_values = np.linspace(0, 2 * np.pi, 1000)
Hf_no_df_values = Hf_no_df(x_values)
Hf_df_values = Hf_df(x_values)

f_values = np.sin(x_values)
plt.plot(x_values, f_values, label='Exact')
plt.plot(x_values, Hf_df_values, label='With derivatives')
plt.plot(x_values, Hf_no_df_values, label='Without derivatives')
plt.legend()
plt.show()
