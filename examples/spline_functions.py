import matplotlib.pyplot as plt

from SEAL.SplineSpace import SplineSpace

p = 2
t = [0, 0, 0, 1, 2, 3, 4, 4, 4]
S = SplineSpace(p, t)

# Parametric
c = [(1, -1, 3), (2, 1, 4), (4, 4, 5), (4, 9, 0), (5, 3, 1), (6, 6, -2)]
f = S(c)
x_values = S.parameter_values(100)
y_values = f(x_values)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(*zip(*y_values))
ax.plot(*zip(*f.control_polygon))
ax.scatter(*zip(*f.control_polygon), s=50)
plt.show()


c = [-1, 1, -1, 1, -1, 1]
f = S(c)
x_values = S.parameter_values(100)
y_values = f(x_values)
poly = f.control_polygon

#
plt.plot(x_values, y_values)
plt.plot(*zip(*poly))
plt.scatter(*zip(*poly))
plt.show()

B = S.basis
for b in B:
    y = b(x_values)
    plt.plot(x_values, y)
    poly = b.control_polygon
    plt.plot(*zip(*poly))
    plt.scatter(*zip(*poly))
plt.show()

print(S)
