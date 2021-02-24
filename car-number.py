import numpy as np
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt

from checkpoints.model_init import model, checkpoint, input_shape
#---------------------------------------
def calc_padding(x, y,w,h, padding):
    x0, y0 = (x + padding, y + padding)
    x1, y1 = (x + w - padding, y + h - padding)
    return x0, y0, x1, y1
# --------------------------------
lower_white=np.array((0,0,0))
upper_white=np.array((60,50,110))
#-----------------------------------
cap = cv2.VideoCapture('번호판야간2.jpg')
_,frame=cap.read()
frame=cv2.resize(frame,(480,300))
#------------------------------------------
M = np.ones(frame.shape, dtype = "uint8") * 50
frame = cv2.add(frame, M)
w,h=frame.shape[:2]
#------------------------------------------
frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,12)
#------------------------------------------

#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#---------------------------------------------------------
white_range=cv2.inRange(hsv,lower_white,upper_white)
white_result=cv2.bitwise_and(frame,frame,mask=white_range)

gray = cv2.cvtColor(white_result, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, (5,5), iterations=13)

cv2.imshow('result',gray)
#---------------------------------------------------------
ret, frame_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('thres result',frame_thresh)
# 1. 빈 프레임 생성
frame_letters = np.zeros(gray.shape, dtype=np.uint8)
# 2. 외곽선 검출
contours, hierachy = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#--------------------------------------------------------
contours_dict = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    # insert to dict
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

#-------------------------------------------------
MIN_AREA = 40
MIN_WIDTH, MIN_HEIGHT = 8, 2
MIN_RATIO, MAX_RATIO = 1.5, 2.5
#--------------------------------------------------
possible_contours = []
cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']
    print('area',area, 'ratio',ratio)
    if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)

for d in possible_contours:
    cv2.rectangle(frame, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(0, 0, 255), thickness=2)

#------------------------------------------------------------
x0, y0, x1, y1 = calc_padding(d['x'], d['y'],d['w'],d['h'] ,25)

frame_letters = cv2.resize(frame_letters, dsize=(x1-x0,y1-y0))
frame_letters = frame[y0:y0+(y1-y0),x0:x0+(x1-x0)]

frame_gray = cv2.cvtColor(frame_letters, cv2.COLOR_BGR2GRAY)

ret, frame_thresh = cv2.threshold(frame_gray, 108, 255, cv2.THRESH_BINARY_INV)
#------------------------------------------------------------
frame_letters = cv2.equalizeHist(frame_thresh)

contours, hierachy = cv2.findContours(frame_letters, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.imshow('33',frame_letters)

contours = sorted(contours, key=lambda cont: cv2.boundingRect(cont)[0])  # 바운딩 박스의 x,y,w,h중 x를 기준으로 sort

x_predict = []

for cont in contours:
    if cv2.contourArea(cont) < 950: continue
    print('dd', cv2.contourArea(cont))
    x, y, w, h = cv2.boundingRect(cont)
    x=x-5
    y=y-5
    w=w+7
    h=h+7
    # 1. 한 문자만을 마스킹하는 이미지 생성
    mask = np.zeros(frame_letters.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cont], 0, 255, -1)

    # 2. 해당 문자만을 잘라냄
    masked = cv2.bitwise_and(frame_letters, frame_letters, mask=mask)[y:y + h, x:x + w]

    # 3. 새로 생성한 정사각형 형태의 프레임에 중앙에 문자 삽입
    n = max([h, w])

    dx, dy = 0, 0  # 잘라낸 숫자를 (dx,dy)만큼 평행이동하여, 중앙에 위치하도록 옮김.
    if (w > h):
        dy = (w - h) // 2
    else:
        dx = (h - w) // 2

    square_frame = np.zeros((n, n), dtype=np.uint8)
    square_frame[dy:dy + h, dx:dx + w] = masked

    # 4. 모델에 입력할 수 있는 형태가 되도록 데이터의 크기를 조정
    _x = cv2.resize(square_frame, (28, 28))
    _x = _x.reshape(input_shape)

    x_predict.append(_x)

# --------------------------------
# 문자 영역으로부터 문자 검출
# --------------------------------
model.load_weights(checkpoint)

x_predict = np.array(x_predict)
y_predict = [np.where(p == max(p))[0][0] for p in model.predict(x_predict)]


# --------------------------------
# 결과 출력
# --------------------------------
#n_of_contours = len(contours)

n_of_contours = 5
plt.figure(figsize=(15, 10))
plt.xlabel((y_predict))
plt.imshow(frame_letters, cmap='gray')

fig, axs = plt.subplots(1, n_of_contours, figsize=(15, 9))
for i in range(n_of_contours):
    axs[i].imshow(np.squeeze(x_predict[i]), cmap='gray')
    axs[i].set_xlabel('predict:' + str(y_predict[i]))

plt.show()
print(y_predict)

cv2.imshow('5',frame_letters)
cv2.waitKey(0)
cv2.destroyAllWindows()
