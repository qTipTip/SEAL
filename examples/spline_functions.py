import matplotlib.pyplot as plt
import numpy as np

from SplineFunction import SplineFunction

p = 1
t = [0, 0, 0, 1, 2, 3, 4, 4, 4]
c = [1, -1, 1, -1, 1, -1, 1]

f = SplineFunction(p, t, c)

x_values = np.linspace(t[0], t[-1], 1000)
y_values = f(x_values)

plt.plot(x_values, y_values)
plt.show()
