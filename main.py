from config import *

cap = cv2.VideoCapture(video_stream_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

while cap.isOpened():
    try:
        success, frame = cap.read()
        if success:
            res = yolo_and_ocr(frame)
            print(res)
        else:
            print("reconnect")
            cap = cv2.VideoCapture(video_stream_path)
    except KeyboardInterrupt:
        break
