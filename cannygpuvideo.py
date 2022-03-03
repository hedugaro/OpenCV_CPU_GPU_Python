import numpy as np
import cv2 as cv
from cv2 import cuda
from matplotlib import pyplot as plt

cap = cv.VideoCapture('sample.mp4')
ret, frame = cap.read()
gpu_frame = cv.cuda_GpuMat()

while (cap.isOpened()):
    cv.imshow('Original video', frame)
    frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    gpu_frame.upload(frame)
    detector = cv.cuda.createCannyEdgeDetector(low_thresh=100, high_thresh=200)
    dstImg = detector.detect(gpu_frame)
    canny = dstImg.download()
    cv.imshow('Canny video', canny)
    if cv.waitKey(25) & 0xFF == ord('e'):
        break
    ret, frame = cap.read()

cap.release()
cv.destroyAllWindows()

