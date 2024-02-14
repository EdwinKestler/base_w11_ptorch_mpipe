import cv2
import os
from screeninfo import get_monitors

RTSP_URL = 'rtsp://admin:FBx!admin2023@192.168.1.108:554'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

# Get screen size from the first monitor
screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height

# Calculate desired window size as 1/4 of the screen size
desired_width = screen_width // 2
desired_height = screen_height // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to desired dimensions
    resized_frame = cv2.resize(frame, (desired_width, desired_height))
    cv2.imshow('RTSP stream', resized_frame)

    if cv2.waitKey(1) == 27:  # ESC key to exit
        break
    if cv2.getWindowProperty('RTSP stream', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()