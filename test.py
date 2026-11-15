import numpy as np
import random


b = np.zeros(2)
b[1] = 1
a = np.concatenate([b] * 10)
print(a)
