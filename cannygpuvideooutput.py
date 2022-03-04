import numpy as np
import cv2 as cv
from cv2 import cuda
from matplotlib import pyplot as plt
import time
import csv

cap = cv.VideoCapture('sample.mp4')
fps = int(cap.get(cv.CAP_PROP_FPS))
fourcc = cv.VideoWriter_fourcc(*"mp4v")
width  = int(cap.get(3))
height = int(cap.get(4))
output = cv.VideoWriter('outputGPU.mp4', fourcc, fps, (width, height),0)
gpu_frame = cv.cuda_GpuMat()

ret, frame = cap.read()

i=0
with open('outputGPU.csv', mode='w', newline='') as output_file:
    while cap.isOpened():
        if ret == True:
            frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            gpu_frame.upload(frame)
            start = time.perf_counter()
            detector = cv.cuda.createCannyEdgeDetector(low_thresh=100, high_thresh=200)
            dstImg = detector.detect(gpu_frame)
            end = time.perf_counter()
            canny = dstImg.download()
            canny = cv.resize(canny, (width, height))
            output.write(canny)


            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            output_writer.writerow([str(i), str(end-start)])
            i=i+1

            ret, frame = cap.read()
        else:
            break

cap.release()
output.release()
cv.destroyAllWindows()
