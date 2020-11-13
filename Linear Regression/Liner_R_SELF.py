from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

from numpy.core.numeric import correlate


style.use('fivethirtyeight')

# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def createDataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def bestFitSlope(xs, ys):
    m = (((mean(xs) * mean(ys)) - (mean(xs*ys))) /
         ((mean(xs) ** 2) - mean(xs ** 2)))
    b = mean(ys) - m*mean(xs)
    return m, b


xs, ys = createDataset(40, 20, 2, correlation='pos')


m, b = bestFitSlope(xs, ys)


def squaredErr(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)


def coefficientDetermination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squaredErrReg = squaredErr(ys_orig, ys_line)
    # y_menaline [meanvalue, meanValue, ..]
    squaredErrMean = squaredErr(ys_orig, y_mean_line)
    return 1 - (squaredErrReg / squaredErrMean)


regression_line = [(m*x)+b for x in xs]


predect_x = 8
predect_y = (m*predect_x + b)

r_sqaure = coefficientDetermination(ys, regression_line)
print(r_sqaure)

plt.scatter(xs, ys)
plt.scatter(predect_x, predect_y, s=100)
plt.plot(xs, regression_line)
plt.show()
