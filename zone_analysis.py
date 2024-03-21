import torch
# !nvcc --version
# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

import os
HOME = os.getcwd()
print(HOME)

model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

import numpy as np
np.bool = np.bool_

OFFICE_SPACE = 'rtsp://192.168.30.3:9000/live'

import datetime
import cv2
import pandas as pd



from queue import Queue
import threading

import cv2
import queue
import threading
import numpy as np
from sshkeyboard import listen_keyboard, stop_listening
import supervision as sv
import time

thread_status = threading.Event()


import cv2
import queue
import threading

# Create a dictionary with the data
data = {
    'current_datetime': [],  # Assuming 'datetime_column' is the name of your datetime column
    'color': [],  # Assuming 'value_column' is the name of your value column
    'count_value': []
}

# RTSP URL for the video stream
rtsp_stream = 'rtsp://192.168.30.3:9000/live'

thread_stop = threading.Event()

colors = sv.ColorPalette.default()

polygons = [
    np.array([
        [925,  275 ],
        [650, 285],
        [330, 1080],
        [925,  1080],
    ], np.int32),
    np.array([
        [928,    510],
        [928,  1080 ],
        [1883,    1080   ],
        [1400, 517]
    ], np.int32),
    np.array([
        [1203, 285 ],
        [1895, 1080],
        [1895, 780],
        [1350, 300]
        # [2160,    0]
    ], np.int32),
    
]
video_info = sv.VideoInfo.from_video_path(rtsp_stream)

zones = [
    sv.PolygonZone(
        polygon=polygon, 
        frame_resolution_wh=video_info.resolution_wh
    )
    for polygon
    in polygons
]
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone, 
        color=colors.by_idx(index), 
        thickness=3,
        text_thickness=4,
        text_scale=4
    )
    for index, zone
    in enumerate(zones)
]
box_annotators = [
    sv.BoxAnnotator(
        color=colors.by_idx(index), 
        thickness=4, 
        text_thickness=4, 
        text_scale=2
        )
    for index
    in range(len(polygons))
]

# Define the number of frames to skip
skip_frames = 1  # Adjust as needed
frame_count = 0
# Create a queue for storing frames
q = queue.Queue()

# Function for receiving frames from the stream and putting them into the queue
def receive_frames(stop):
    cap = cv2.VideoCapture(rtsp_stream)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                q.put(frame)
            if stop() | ret == False:
                break
            # else:
            #     break
            
    cap.release()

def process_frame(frame: np.ndarray) -> np.ndarray:

    # global img
    # img = frame
    # detect
    results = model(frame, size=800)
    detections = sv.Detections.from_yolov5(results)
    detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]

    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
        frame = zone_annotator.annotate(scene=frame)
        
        # Get the current date and time
        if zone.current_count >= 3:
            current_datetime = datetime.datetime.now()
            data['current_datetime'].append(current_datetime)
            data['color'].append(zone_annotators.index(zone_annotator))
            data['count_value'].append(zone.current_count)
        # print(zone_annotators.index(zone_annotator))
        # print(zone.current_count)

    return frame

stop_threads = False
# Start the background thread for receiving frames
receive_thread = threading.Thread(target=receive_frames, args =(lambda : stop_threads, ))
receive_thread.start()

time.sleep(1)
# Main loop for displaying frames
while True:
    # Check if there are frames in the queue
    try:
        frame = q.get(timeout=0.001)
        frame_count += 1
    except queue.Empty:
        frame = None

    if frame is not None:
        # Check if it's time to process this frame
        if frame_count % (skip_frames + 1) == 0:
            frame = process_frame(frame)
            # Display the frame
            cv2.imshow('Frame', cv2.resize(frame, (800,600)))

        # Poll for key events
    key = cv2.waitKey(1)

    # Check for the 'q' key press event to exit
    if key & 0xFF == ord('q'):
        thread_stop.set()
        stop_threads = True
        del q
        receive_thread.join()

        break

# Release resources and close windows
cv2.destroyAllWindows()
