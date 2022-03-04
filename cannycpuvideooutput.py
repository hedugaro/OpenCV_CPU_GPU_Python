import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import csv

cap = cv.VideoCapture('sample.mp4')
fps = int(cap.get(cv.CAP_PROP_FPS))
fourcc = cv.VideoWriter_fourcc(*"mp4v")
width  = int(cap.get(3))
height = int(cap.get(4))
output = cv.VideoWriter('outputCPU.mp4', fourcc, fps, (width, height),0)

i=0
with open('outputCPU.csv', mode='w', newline='') as output_file:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            start = time.perf_counter()
            edge_detect = cv.Canny(frame, 100, 200)
            end = time.perf_counter()
            edge_detect = cv.resize(edge_detect, (width, height))
            output.write(edge_detect)

            
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            output_writer.writerow([str(i), str(end-start)])
            i=i+1        
        else:
            break

cap.release()
output.release()
cv.destroyAllWindows()
