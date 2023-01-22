import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

x = np.random.randn(10)
y = np.random.randn(10)
Cluster = np.array([0, 1, 1, 1, 3, 2, 2, 3, 0, 2])

fig = plt.figure()
ax = fig.add_subplot(111)

scatter = ax.scatter(x, y, s=50)
for i, j in centers:
   ax.scatter(i, j, s=50, c='red', marker='+')

plt.show()