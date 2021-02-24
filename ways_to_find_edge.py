import cv2
import numpy as np

frame = cv2.imread('images/3.jpg')
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

ddepth = 5
sobelX = cv2.Sobel(frame_gray, ddepth, 1, 0)
sobelY = cv2.Sobel(frame_gray, ddepth, 0, 1)
edge_raw = cv2.add(sobelX, sobelY)

sobel_edge = cv2.convertScaleAbs(edge_raw)
threshed = cv2.threshold(sobel_edge, 128, 255, cv2.THRESH_BINARY)[-1]

canny_edge = cv2.Canny(frame_gray, 0, 128)

cv2.imshow('sobel', sobel_edge)
cv2.imshow('canny', canny_edge)
cv2.imshow('threshed', threshed)
cv2.waitKey(0)
cv2.destroyAllWindows()