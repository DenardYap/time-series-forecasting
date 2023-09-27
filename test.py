import numpy as np

# Your array

a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(a.shape, b.shape)

print(a[:,np.newaxis].shape)

print(a * b[:, np.newaxis])


# 10 = dimension
# 5 = class 
