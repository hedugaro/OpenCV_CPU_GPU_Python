import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture('sample.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()
    cv.imshow('Original video', frame)
    edge_detect = cv.Canny(frame, 100, 200)
    cv.imshow('Canny video', edge_detect)
    if cv.waitKey(25) & 0xFF == ord('e'):
        break

cap.release()
cv.destroyAllWindows()
