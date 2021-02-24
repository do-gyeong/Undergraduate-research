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
                (128, 128, 255),  # Font color
                lineType=cv2.LINE_AA)



cv2.namedWindow('Binary')
capture = cv2.VideoCapture("number.mp4")
#capture = cv2.VideoCapture(0)

lower_yellow = np.array([0, 0, 0])
upper_yellow = np.array([20, 255, 255])

cv2.createTrackbar('Threshold','Binary',0,255,nothing)
cv2.setTrackbarPos('Threshold','Binary',140)

model = tf.keras.models.Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --------------------------------
# mnist 데이터베이스를 이용한 학습
# --------------------------------

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model.fit(x_train, y_train, epochs=12)
model.evaluate(x_test, y_test, verbose=2)

while True:
    ret, frame = capture.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    yellow_range = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # 두 이미지에서 모두 mask값에 해당하는 부분만 저장
    yellow_result = cv2.bitwise_and(frame, frame, mask=yellow_range)
    yellow_result = cv2.resize(yellow_result, (300, 300))

    threshold = cv2.getTrackbarPos('Threshold','Binary')
    ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary', binary)

    # 해당 Binary 이미지를 반전시킴
    binary = cv2.bitwise_not(binary)
    contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # connectedComponentsWithStat() : Python 3.0 부터 생긴 라벨링 함수
    # numofLabels : 레이블 개수 (반환값이 N인 경우 0~N-1까지의 레이블 번호 존재)
    # img_label : 레이블링된 입력 영상과 같은 크기의 배열
    # stats : N x 5 행렬(N:레이블 개수), [x좌표,y좌표,폭,높이,넓이]
    # centroids :  각 레이블의 중심점 좌표, N X 2 행렬
    # yellow_range : 입력 영상
    print('')
    numOfLabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(yellow_range)

    # enumerate(순서가 있는 자료형) : 리스트의 순서와 리스트의 값을 전달하는 기능
    # 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 리턴
    # idx : Index 번호
    # centroid : 해당 Index에 대한 중심점 좌표
    for idx in range(len(centroids)):

        x, y, width, height = cv2.boundingRect(contours[idx])
        # 중심좌표 값
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255),2)
    print('centroids 수',len(contours))

    for i in range(len(contours)):
        x, y, width, height = cv2.boundingRect(contours[i])
        frame1 = frame[x,x+width,y,y+height]
        frame1 = cv2.resize(frame, (448, 448))

        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        yellow_range = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # 두 이미지에서 모두 mask값에 해당하는 부분만 저장
        yellow_result = cv2.bitwise_and(frame1, frame1, mask=yellow_range)
        yellow_result = cv2.resize(yellow_result, (28, 28))

        x = np.expand_dims(yellow_result.reshape(28,28), axis=0)
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
            model.fit(x_train, y_train, epochs=1)
        writeText(frame, "Predict: %d, Label : %s" % (y, str(y_train[0])))
        frame1[:28, :28] = cv2.cvtColor(x[0], cv2.COLOR_GRAY2BGR)

    cv2.imshow("original", frame)
    cv2.imshow('Camera', frame1)
    cv2.imshow("yellow_line_detection", yellow_result)

    # 이미지를 갱신하기 위해 waitKey()를 이용해 50ms만큼 대기한 후 다음 프레임으로 넘어감
    # q가 입력되면 동영상 재생 중지 -> Python OpenCV는 문자를 처리하지 못하므로 유니코드 값으로 변환하기 위해 ord() 사용
    if cv2.waitKey(50) == ord('q'): break

# 동영상 재생이 끝난 후 동영상 파일을 닫고 메모리를 해제
capture.release()
# 윈도우를 닫음
cv2.destroyAllWindows()
