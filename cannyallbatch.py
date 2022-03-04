import numpy as np
import cv2 as cv
from cv2 import cuda
from matplotlib import pyplot as plt
import os
import time

L1=[]
L2=[]
Li=[]
i=1
your_path = 'cat/'
files = os.listdir(your_path)
for file in files:
    if os.path.isfile(os.path.join(your_path, file)):
        img = cv.imread(os.path.join(your_path, file),0)
        start = time.time()
        canny = cv.Canny(img,100,110)
        end = time.time()
        L1.append(end-start)
        Li.append(i)
        i=i+1
        cv.imwrite(os.path.join(your_path+"/outputCPU/", file), canny)
        print(str(file))


your_path = 'cat/'
files = os.listdir(your_path)
for file in files:
    if os.path.isfile(os.path.join(your_path, file)):
        img = cv.imread(os.path.join(your_path, file),0)
        imgMat = cv.cuda_GpuMat(img)
        detector = cv.cuda.createCannyEdgeDetector(low_thresh=100, high_thresh=110)
        start2 = time.time()
        dstImg = detector.detect(imgMat)
        end2 = time.time()
        L2.append(end2-start2)
        canny = dstImg.download()
        cv.imwrite(os.path.join(your_path+"/outputGPU/", file), canny)


plt.plot(Li, L1, '.', color = 'g', label = "CPU time")
plt.plot(Li, L2, '.', color = 'r', label = "GPU time")
plt.xticks(rotation = 25)
plt.xlabel('Frame')
plt.ylabel('Time')
plt.title('CPU vs GPU', fontsize = 20)
plt.grid()
plt.legend()
plt.show()
