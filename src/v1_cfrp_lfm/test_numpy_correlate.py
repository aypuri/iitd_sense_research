import cv2 as cv
import numpy as np

# initialize a np array with numbers ranging from 1 to 20
a = np.arange(20)
print(a)
# initialize a np array with 20 2s
b = np.full(20, 2)
print(b)
c = np.correlate(a, b, "same")
print(c)