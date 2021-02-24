import cv2
import numpy as np
import pytesseract
from PIL import Image
def calc_padding(x, y,w,h, padding):
    x0, y0 = (x + padding, y + padding)
    x1, y1 = (x + w - padding, y + h - padding)
    return x0, y0, x1, y1

img = cv2.imread('Sample1.jpg')
img = cv2.fastNlMeansDenoisingColored(img,None,7,7,5,7)
img_height,img_width = img.shape[:2]
Number = img

cv2.imshow('Number', Number)
copy_img = img.copy()
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel=np.ones((1,1),np.uint8)
img2=cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernel)
#cv2.imwrite('gray.jpg', img2)
blur = cv2.GaussianBlur(img2, (3, 3), 0)
#cv2.imwrite('blur.jpg', blur)

ret, img_thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
#cv2.imwrite('canny.jpg', img_thresh)
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

number=[]
box1 = []
rect = []
sort = []
index = []
xx = []
yy = []
hh = []
ww = []

f_count = 0
select = 0
plate_width = 0
img_thresh1 = np.zeros(img_thresh.shape, dtype=np.uint8)
img2 = np.zeros(img_thresh.shape, dtype=np.uint8)
img3 = np.zeros(img_thresh.shape, dtype=np.uint8)

for i in range(len(contours)):
    cnt = contours[i]
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    rect.append(cv2.boundingRect(cnt))
    rect_area = w * h  # area size
    aspect_ratio = float(w) / h  # ratio = width/height

    if (aspect_ratio >= 3) and (aspect_ratio <= 5) and (rect_area >= 500) :

        img_thresh1=img_thresh[y:y+h,x:x+w]
        img2=img[y:y+h,x:x+w]

        contours1, hierarchy = cv2.findContours(img_thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours1)):
            cnt1 = contours1[i]
            area1 = cv2.contourArea(cnt1)
            x1, y1, w1, h1 = cv2.boundingRect(cnt1)
            rect_area1 = w1 * h1  # area size
            aspect_ratio1 = float(w1) / h1  # ratio = width/height
            if (aspect_ratio1 >= 0.2) and (aspect_ratio1 <= 1.0) and (rect_area1 >= 270) and (rect_area1 <= 700):
                cv2.rectangle(img2, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
                img3=img2.copy()
                box1.append(cv2.boundingRect(cnt1))
                xx.append(x1)
                yy.append(y1)
                ww.append(w1)
                hh.append(h1)
cv2.waitKey(0)
cv2.imshow('BoundingBox',img3)
cv2.waitKey(0)
minx=min(xx)
maxx=max(xx)
miny=min(yy)
maxy=max(yy)
maxw=max(ww)
maxh=max(hh)

copy1 = img3.copy()

copy1=copy1[miny:maxy+maxh,minx:maxx+maxw]
cv2.imshow('N',copy1)
copy1=cv2.copyMakeBorder(copy1,50,50,50,50,cv2.BORDER_CONSTANT,value=[255,255,255])
copy1 = cv2.cvtColor(copy1, cv2.COLOR_BGR2GRAY)
ret, copy1 = cv2.threshold(copy1, 120, 255, cv2.THRESH_BINARY)

copy1=cv2.morphologyEx(copy1,cv2.MORPH_CLOSE,(5,5))
cv2.waitKey(0)
cv2.imshow('result',copy1)
cv2.waitKey(0)

result = pytesseract.image_to_string(copy1, lang="kor")
print(result)

cv2.destroyAllWindows()
cv2.imshow('Original',img)
cv2.waitKey(0)
