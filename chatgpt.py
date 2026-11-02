
import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model',
    help='Path to YOLO model file (example: "best.pt")',
    required=True
)

parser.add_argument(
    '--source',
    help='Image file, image folder, video file, or IP camera RTSP URL',
    required=True
)

parser.add_argument(
    '--thresh',
    help='Minimum confidence threshold (example: 0.4)',
    default=0.5
)

parser.add_argument(
    '--resolution',
    help='Display resolution in WxH (example: 640x480)',
    default=None
)

parser.add_argument(
    '--record',
    help='Record detection results and save as demo1.avi',
    action='store_true'
)

args = parser.parse_args()

# Parse user inputs

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check if model exists

if not os.path.exists(model_path):
    print('ERROR: Model file not found.')
    sys.exit(0)

# Load YOLO model

model = YOLO(model_path, task='detect')
labels = model.names

# Supported extensions

img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                '.png', '.PNG', '.bmp', '.BMP']

vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

# Identify source type

if os.path.isdir(img_source):
    source_type = 'folder'

elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)

    if ext in img_ext_list:
        source_type = 'image'

    elif ext in vid_ext_list:
        source_type = 'video'

    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)

elif img_source.startswith(('rtsp://', 'rtsps://')):
    source_type = 'ip_camera'

else:
    print('Invalid input source.')
    sys.exit(0)

# Parse resolution

resize = False

if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Recording setup

if record:

    if source_type not in ['video', 'ip_camera']:
        print('Recording only works with video and IP camera sources.')
        sys.exit(0)

    if not user_res:
        print('Please specify resolution using --resolution.')
        sys.exit(0)

    record_name = 'demo1.avi'
    record_fps = 30

    recorder = cv2.VideoWriter(
        record_name,
        cv2.VideoWriter_fourcc(*'MJPG'),
        record_fps,
        (resW, resH)
    )

# Load image or video source

if source_type == 'image':
    imgs_list = [img_source]

elif source_type == 'folder':
    imgs_list = []

    filelist = glob.glob(img_source + '/*')

    for file in filelist:
        _, file_ext = os.path.splitext(file)

        if file_ext in img_ext_list:
            imgs_list.append(file)

elif source_type in ['video', 'ip_camera']:

    if source_type == 'video':
        cap_arg = img_source

    else:
        cap_arg = img_source
        print('Connecting to IP Camera...')

    cap = cv2.VideoCapture(cap_arg, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print('ERROR: Cannot connect to video source.')
        sys.exit(0)

    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

    if source_type == 'ip_camera':
        print('IP Camera connected successfully!')

# Bounding box colors

bbox_colors = [
    (164,120,87), (68,148,228), (93,97,209),
    (178,182,133), (88,159,106), (96,202,231),
    (159,124,168), (169,162,241), (98,118,150),
    (172,176,184)
]

# Initialize variables

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop

while True:

    t_start = time.perf_counter()

    # Read image or frame

    if source_type in ['image', 'folder']:

        if img_count >= len(imgs_list):
            print('All images have been processed.')
            break

        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1

        if frame is None:
            print('Unable to read image.')
            continue

    elif source_type in ['video', 'ip_camera']:

        ret, frame = cap.read()

        if not ret or frame is None:

            if source_type == 'video':
                print('Reached end of video.')
                break

            else:
                print('Unable to receive IP camera frame.')
                break

    # Resize frame

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # YOLO inference

    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Object counting

    object_count = 0

    for i in range(len(detections)):

        xyxy = detections[i].xyxy.cpu().numpy().squeeze()

        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        conf = detections[i].conf.item()

        if conf > min_thresh:

            color = bbox_colors[classidx % 10]

            cv2.rectangle(
                frame,
                (xmin, ymin),
                (xmax, ymax),
                color,
                2
            )

            label = f'{classname}: {int(conf * 100)}%'

            labelSize, baseLine = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            label_ymin = max(ymin, labelSize[1] + 10)

            cv2.rectangle(
                frame,
                (xmin, label_ymin-labelSize[1]-10),
                (xmin+labelSize[0], label_ymin+baseLine-10),
                color,
                cv2.FILLED
            )

            cv2.putText(
                frame,
                label,
                (xmin, label_ymin-7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

            object_count += 1

    # Display FPS

    if source_type in ['video', 'ip_camera']:

        cv2.putText(
            frame,
            f'FPS: {avg_frame_rate:.2f}',
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # Display object count

    cv2.putText(
        frame,
        f'Number of chickens: {object_count}',
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Show results

    cv2.imshow('YOLO IP Camera Detection', frame)

    if record:
        recorder.write(frame)

    # Keyboard controls

    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        cv2.waitKey()

    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

    # Calculate FPS

    t_stop = time.perf_counter()

    frame_rate_calc = 1 / (t_stop - t_start)

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)

    frame_rate_buffer.append(frame_rate_calc)

    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up

print(f'Average pipeline FPS: {avg_frame_rate:.2f}')

if source_type in ['video', 'ip_camera']:
    cap.release()

if record:
    recorder.release()

cv2.destroyAllWindows()
