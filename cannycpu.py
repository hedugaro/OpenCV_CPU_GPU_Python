import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('beetle.png',0)
canny = cv.Canny(img,100,110)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny,cmap = 'gray')
plt.title('Canny Image'), plt.xticks([]), plt.yticks([])
plt.show()
