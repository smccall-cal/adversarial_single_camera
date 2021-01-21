#!/bin/src/python

import numpy as np
from matplotlib import pyplot

from payoff_matrix import PayoffMatrix

floormap = (1 - pyplot.imread("resources/bitmap.png")[::10, ::10, 0])
strategy = np.ones(floormap.shape) / np.prod(floormap.shape)

matrix = PayoffMatrix(floormap)
payoff = matrix.expected_payoff(strategy)

pyplot.imshow(1 - floormap, cmap="gray")
pyplot.show()

pyplot.imshow(payoff, cmap="gray")
pyplot.show()