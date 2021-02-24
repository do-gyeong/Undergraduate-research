import numpy as np
import cv2
import datetime

# VideoCapture(method)
# method : 전달 인자 -> File Name 혹은 Device
# Device : 연결된 영상 장치 (연결 할 Device가 1개인 경우 0)
capture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = None

# 수정한부분: 녹화 설정
path = "./"  # "C:/" # 파일이 저장될 경로
record = False  # 녹화 여부

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

# 수정한부분: 녹화 관련 함수
isYellowDetected = False


# 수정한부분: yellow_detected: 노랑이 검출되었을 때 실행해줘야 하는 함수
def yellow_detected():
    global isYellowDetected

    if not isYellowDetected:
        start_record()
    else:
        video.write(frame)

    isYellowDetected = True


# 수정한부분: yellow_undetected: 노랑이 더 이상 검출되지 않을 때 실행해줘야 하는 함수
def yellow_undetected():
    global isYellowDetected

    if isYellowDetected:
        stop_record()

    isYellowDetected = False


# 수정한부분: start_record: 녹화 시작
def start_record():
    global video
    fname = path + "%s.avi" % str(now)
    video = cv2.VideoWriter(fname, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    print("녹화 시작 @[%s]... " % fname, end="")


# 수정한부분: stop_record: 녹화 중지
def stop_record():
    global video
    if not video is None:
        video.release()
        print("저장됨")


def record(frame):
    if record:
        video.write(frame)


while True:
    ret, frame = capture.read()

    frame = cv2.resize(frame, (256, 256))  # 수정한부분: 정규화
    frame = cv2.GaussianBlur(frame, (3, 3), 25)  # 수정한부분

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    yellow_range = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # 두 이미지에서 모두 mask값에 해당하는 부분만 저장
    yellow_result = cv2.bitwise_and(frame, frame, mask=yellow_range)
    # yellow_result = cv2.resize(yellow_result, (300, 300)) # 수정한부분: 삭제

    ret, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)  # 수정한부분
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, (3, 3), iterations=3)  # 수정한부분

    contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # connectedComponentsWithStat() : Python 3.0 부터 생긴 라벨링 함수
    # numofLabels : 레이블 개수 (반환값이 N인 경우 0~N-1까지의 레이블 번호 존재)
    # img_label : 레이블링된 입력 영상과 같은 크기의 배열
    # stats : N x 5 행렬(N:레이블 개수), [x좌표,y좌표,폭,높이,넓이]
    # centroids :  각 레이블의 중심점 좌표, N X 2 행렬
    # yellow_range : 입력 영상
    numOfLabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(yellow_range)
    # enumerate(순서가 있는 자료형) : 리스트의 순서와 리스트의 값을 전달하는 기능
    # 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 리턴
    # idx : Index 번호
    # centroid : 해당 Index에 대한 중심점 좌표
    counter_yellowDetected = 0  # 수정한부분
    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:  # x,y좌표가 0일 경우 다음 Index로 넘어감
            continue

        # np.isnan(array 객체) : 객체의 원소단위로 판단하여 각 원소가 NAN(Not A Number)일 경우 True반환
        # NAN : 연산 과정에서 잘못된 입력을 받았음을 나타내는 기호
        # -> 해당 원소가 NAN인 경우 다음 Index로 넘어감
        if np.any(np.isnan(centroid)):
            continue

        # 리턴값으로 테두리 그리기를 위한 좌표값
        x, y, width, height, area = stats[idx]
        # 중심좌표 값
        centerX, centerY = int(centroid[0]), int(centroid[1])

        # 노이즈로 인해 검출된 작은 물체 제거를 위해 크기가 50이상인것만 검출
        if area > 32:  # 수정한부분: 값 조정
            now = datetime.datetime.now().strftime("%d_%H-%M-%S")
            # circle() : 중심점 좌표를 원으로 그림
            # frame : 원이 그려질 이미지
            # (centerX, centerY) : 원의 중심 좌표
            # 10 : 원의 반지름
            # (0, 0, 255) : 원의 색(B, G, R)
            # 10 : 선의 굵기
            cv2.circle(frame, (centerX, centerY), 5, (0, 0, 255), 10)
            # rectangle() : 검출된 Line에 사각형 윤곽선 그림
            # frame : 사각형이 그려질 이미지
            # (x, y) : 사각형의 시작점
            # (x + width, y + height) : 시작점과 대각선에 있는 사각형의 끝점
            # (0, 0, 255) : 사각형의 색(B, G, R)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255))
            counter_yellowDetected += 1  # 수정한부분

    if counter_yellowDetected > 0:  # 수정한부분
        yellow_detected()
        record(frame)
    else:
        yellow_undetected()

    cv2.imshow("original", frame)
    cv2.imshow("yellow_line_detection", yellow_result)

    # 이미지를 갱신하기 위해 waitKey()를 이용해 50ms만큼 대기한 후 다음 프레임으로 넘어감
    # q가 입력되면 동영상 재생 중지 -> Python OpenCV는 문자를 처리하지 못하므로 유니코드 값으로 변환하기 위해 ord() 사용
    key = cv2.waitKey(50) & 0xFF  # 수정한부분
    if key == ord('q'):
        break

# 동영상 재생이 끝난 후 동영상 파일을 닫고 메모리를 해제
capture.release()
stop_record()
cv2.destroyAllWindows()  # 윈도우를 닫음
