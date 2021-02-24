import cv2
import numpy as np

frame = cv2.imread('images/3.jpg')
dst = frame.copy()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

ddepth = 5
sobelX = cv2.Sobel(frame_gray, ddepth, 1, 0)
sobelY = cv2.Sobel(frame_gray, ddepth, 0, 1)
edge_raw = cv2.add(sobelX, sobelY)

sobel_edge = cv2.convertScaleAbs(edge_raw)
threshed = cv2.threshold(sobel_edge, 170, 255, cv2.THRESH_BINARY)[-1]
threshed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, (3,3), iterations=3)
contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

frame1=frame
frame1= cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
cv2.imshow('frame1', frame1)
for i in range(len(contours)):
    cnt = contours[i]
    area = cv2.contourArea(cnt)
    rect = cv2.minAreaRect(cnt)
    print(rect)
    box_area = rect[1][0] * rect[1][1]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    frame = cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

    epsilon = 0.001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    size = len(approx)
    cv2.drawContours(dst, [approx], 0, (0, 255, 0 ), 2)


cv2.imshow('dst4', dst)
cv2.imshow('frame', frame)
cv2.imshow('threshed', threshed)
cv2.waitKey(0)
cv2.destroyAllWindows()