--------------------------------------
2020-05-14 에 작성한 파일로 mnist 손글씨 인식에 대한 코드입니다.
-----------------------------------------

import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
def nothing(x):
    pass

def getNumericKey(key):
    # 0~9사이의 키가 눌렸을 때만 해당 키를 반환.
    # 그 외의 키가 눌리면 None을 반환한다.
    c = chr(key)
    if c in "0123456789":
        return int(c)
    return None

def writeText(frame1, txt):
    # 화면 프레임에 글씨 쓰기
    frame1[:28, :] = frame1[:28, :] // 2
    cv2.putText(frame1,
                txt,
                (32, 16),  # Coordinates
                cv2.FONT_HERSHEY_PLAIN,  #
                1.2,  # Font scale
                (0, 255,0 ),  # Font color
                lineType=cv2.LINE_AA)

#cv2.namedWindow('Binary')
capture = cv2.VideoCapture('final.mp4')
#capture = cv2.VideoCapture(0)

lower_yellow = np.array([0, 0, 0])
upper_yellow = np.array([20, 255, 255])

cv2.namedWindow('Binary') # 수정한 부분
cv2.createTrackbar('Threshold','Binary',0,255,nothing)
cv2.setTrackbarPos('Threshold','Binary',91)

mnist = tf.keras.datasets.mnist

model = tf.keras.models.Sequential([
    Conv2D(filters=32,
           kernel_size=(3, 3),
           padding='same',
           activation='relu',
           input_shape=(28, 28,1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(filters=64,
           kernel_size=(3, 3),
           padding='same',
           activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=128,
           kernel_size=(3, 3),
           padding='same',
           activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=128,
           kernel_size=(3, 3),
           padding='same',
           activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.7),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.1),
    Dense(10, activation='softmax')
])
#np.squeeze(Flatten, 3)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --------------------------------
# mnist 데이터베이스를 이용한 학습
# --------------------------------

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255.0 # 수정한 부분
x_test  = x_test.reshape( x_test.shape[0], 28, 28, 1) / 255.0 # 수정한 부분

#model.fit(x_train, y_train, batch_size=16, epochs=2) # 잠시 수정
#model.save_weights('tmp/dok') # 잠시 수정
model.load_weights('tmp/dok') # 잠시 수정
model.evaluate(x_test, y_test, verbose=2)

while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = cv2.getTrackbarPos('Threshold','Binary')
    ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    #binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
    #                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
    # 해당 Binary 이미지를 반전시킴
    binary = cv2.bitwise_not(binary)
    contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.imshow('binary',binary)
    for i in range(len(contours)):
        x, y, width, height = cv2.boundingRect(contours[i])
        if width*height >500:

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
            frame1 = frame[ y:y + height,x:x + width]
            frame1 = cv2.resize(frame1, (448, 448))

            hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            yellow_range = cv2.inRange(hsv, lower_yellow, upper_yellow)
            # 두 이미지에서 모두 mask값에 해당하는 부분만 저장
            yellow_result = cv2.bitwise_and(frame1, frame1, mask=yellow_range)

            yellow_result = cv2.resize(yellow_result, (28, 28))

            yellow_result = cv2.cvtColor(yellow_result, cv2.COLOR_BGR2GRAY)
            cv2.imshow('yellow',yellow_result)

            x = tf.expand_dims(yellow_result.reshape(28,28,1), axis=0) # 수정한부분
            p = model.predict(x)[0]  # 학습된 모델을 사용하여 결과 예측

            y = np.where(p == max(p))[0][0]  # one-hot 인코딩에서 0~9사이의 숫자로 변환

            # -----------------------------------------------------------
            # Key mappings
            # -----------------------------------------------------------
            key = cv2.waitKey(20) & 0xFF
            if key == 27: break  # ESC

            # -----------------------------------------------------------
            # 모델의 예측 결과가 부정확 할 시, 재훈련
            # (재훈련에 사용할 레이블은 키보드의 숫자키 0~9로부터 입력받음)
            # -----------------------------------------------------------
            x_train = x
            y_train = np.expand_dims(getNumericKey(key), axis=0)

            if y_train[0] != None:  # 레이블이 주어지면 학습
                model.fit(x_train, y_train,batch_size=32, epochs=1)
            writeText(frame1, "Predict: %d, Label : %s" % (y, str(y_train[0])))

            cv2.putText(frame,str(y),tuple(contours[i][0][0]),cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
            cv2.imshow('before frame1', frame1)
        
            frame1[:28, :28] = cv2.cvtColor(np.squeeze(x[0]), cv2.COLOR_GRAY2BGR) # 수정한 부분
        else :
            cv2.waitKey(1)

    print(model.summary())
    cv2.imshow("original", frame)
    # 이미지를 갱신하기 위해 waitKey()를 이용해 50ms만큼 대기한 후 다음 프레임으로 넘어감
    # q가 입력되면 동영상 재생 중지 -> Python OpenCV는 문자를 처리하지 못하므로 유니코드 값으로 변환하기 위해 ord() 사용
    if cv2.waitKey(1) == ord('q'): break

# 동영상 재생이 끝난 후 동영상 파일을 닫고 메모리를 해제
capture.release()
# 윈도우를 닫음
cv2.destroyAllWindows()
