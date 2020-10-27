import matplotlib.pyplot as plt
from Drop_Recognition_2.video import Video
from Drop_Recognition_2.image_analysis import get_peak_analysis_of_line
from Drop_Recognition_2.image_analysis import get_peak_detection_image
import cv2
import numpy as np
import timeit

# Vars
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"
rain_detected_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Created/rain_detected_video_1"
frame_count = 50
rain_detected_video = []
processing_time = 0
a = None

# Create Video and store it
cap = cv2.VideoCapture(rain_video_path)
for i in range(frame_count):
    ret, frame = cap.read()
    a = [len(frame), len(frame[0])]
    time_before = timeit.default_timer()
    rain_detected_video.append(get_peak_detection_image(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    processing_time += timeit.default_timer() - time_before
    print(timeit.default_timer() - time_before)

print(processing_time/frame_count)
print(a)




