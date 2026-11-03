import cv2

RTSP_URL = "rtsp://admin:saisaicandel23@192.168.1.239:554/Preview_01_sub"

cap = cv2.VideoCapture(RTSP_URL)

if cap.isOpened():
    print("Camera connected successfully!")

    ret, frame = cap.read()

    if ret:
        print("Video frame received successfully!")
        cv2.imwrite("camera_test.jpg", frame)
    else:
        print("Connected, but unable to receive video.")

else:
    print("Failed to connect to camera.")

cap.release()
