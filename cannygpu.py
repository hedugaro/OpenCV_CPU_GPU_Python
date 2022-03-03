import numpy as np
import cv2 as cv
from cv2 import cuda
from matplotlib import pyplot as plt

img = cv.imread('beetle.png',0)
imgMat = cv.cuda_GpuMat(img)
detector = cv.cuda.createCannyEdgeDetector(low_thresh=100, high_thresh=110)
dstImg = detector.detect(imgMat)
canny = dstImg.download()
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
