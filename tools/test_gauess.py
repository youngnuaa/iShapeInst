import cv2
import numpy as np



a = np.zeros(shape=(25, 25))


a[12, 12] = 1

c = cv2.GaussianBlur(a, (7, 7), 7)


print(c)