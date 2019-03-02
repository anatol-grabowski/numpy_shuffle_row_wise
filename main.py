from numpy_shuffle_row_wise import shuffle_row_wise
import numpy as np
from time import time

a = np.arange(5).reshape(1, -1).repeat(100000, axis=0)
print(a)

t = time()
shuffle_row_wise(a)
print(time() - t, 'time')
print(a)
