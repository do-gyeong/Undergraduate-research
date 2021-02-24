import cv2
import numpy as np

def setLabel(image, str, contour):
    (text_width, text_height), baseline = cv2.getTextSize(str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    x,y,width,height = cv2.boundingRect(contour)
    pt_x = x+int((width-text_width)/2)
    pt_y = y+int((height + text_height)/2)
    cv2.rectangle(image, (pt_x, pt_y+baseline), (pt_x+text_width, pt_y-text_height), (200,200,200), cv2.FILLED)
    cv2.putText(image, str, (pt_x, pt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, 8)

img = cv2.imread('images/3.jpg')
cv2.imshow('original img', img)

img = cv2.fastNlMeansDenoisingColored(img,None,7,7,5,7)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray img', img2)

kernel=np.ones((1,1),np.uint8)
img2=cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernel)
cv2.imshow('morphology img', img2)

img2 = cv2.GaussianBlur(img2, (3, 3), 0)


ret, img_thresh = cv2.threshold(img2, 110, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('contour',img_thresh)
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img,contours,-1,(0,255,0),1)


for i in range(len(contours)):
    cnt = contours[i]

    area = cv2.contourArea(cnt)
    rect = cv2.minAreaRect(cnt)
    print(rect)
    box_area = rect[1][0]*rect[1][1]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    epsilon = 0.001 * cv2.arcLength(cnt,True)
    approx=cv2.approxPolyDP(cnt,epsilon,True)

    size=len(approx)
    
    for k in range(size-1):
        #if(abs(box_area - area) <20 ):
            cv2.line(img, tuple(approx[k][0]), tuple(approx[k+1][0]), (255, 0, 0), 1)

    if cv2.isContourConvex(approx):
        if (size == 4):
            #setLabel(img,"rectangle",cnt)
            img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)


cv2.imshow('contour',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

