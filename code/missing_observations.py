import numpy as np
import matplotlib.pyplot as plt
import itertools


# define the matplotlib function to be used
def plot(img):
    plt.figure()
    plt.contourf(img, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.show()
    return


# generate data either complete or with missing observations
def generate_data(missing=False, miss_variance=1e100, style='parabola'):
    if style == 'plane':
        x1 = np.arange(-25, 25, 1)
        x2 = np.arange(-25, 25, 1)
        miss = None
        # create observations on the grid and add some noise. Right now using sine
        X, Y = np.meshgrid(x1, x2)
        size = Y.shape[0] * Y.shape[1]
        z = X + Y * 1.0
    if style == 'parabola':
        # declare a grid
        x1 = np.arange(-25, 25, 1)
        x2 = np.arange(-25, 25, 1)
        miss = None
        # create observations on the grid and add some noise. Right now using sine
        X, Y = np.meshgrid(x1, x2)
        size = Y.shape[0] * Y.shape[1]
        z = X ** 2 / 4 + Y ** 2 / 8
    if style == 'double':
        # declare a grid
        x1 = np.arange(-25, 25, 1) / 100
        x2 = np.arange(-25, 25, 1) / 100
        miss = None
        # create observations on the grid and add some noise. Right now using sine
        X, Y = np.meshgrid(x1, x2)
        size = Y.shape[0] * Y.shape[1]
        z = X ** 3 + Y ** 3
    if style == 'sine':
        # declare a grid
        x1 = np.arange(-25, 25, 1)
        x2 = np.arange(-25, 25, 1)
        miss = None
        # create observations on the grid and add some noise. Right now using sine
        X, Y = np.meshgrid(x1, x2)
        size = Y.shape[0] * Y.shape[1]
        z = np.sin(X) + Y
    if missing:
        points = np.reshape(np.random.uniform(-1, 1, 50 * 50), (50, 50))
        miss = points > 0.90
        points = np.sum(miss)
        print(miss.shape, z.shape)
        z[miss] = np.random.normal(0, miss_variance, points)
        print(np.sum(miss) / size * 100)
    else:
        miss = np.array(np.zeros(z.shape), dtype='bool')
    return z, miss
