from SEAL import SplineSpace
import matplotlib.pyplot as plt

p = 2
t = [0, 0, 0, 1, 2, 2, 2]
S = SplineSpace(p, t)

c = [1, -1, 1, -3]
f = S(c)

for i in range(10):
    cp = f.visualize(iterations=i)
    plt.plot(*zip(*cp), label="{}".format(i))
    plt.legend()
    plt.show()