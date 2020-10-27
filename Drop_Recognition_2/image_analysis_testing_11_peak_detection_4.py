import matplotlib.pyplot as plt
from Drop_Recognition_2.video import Video
from Drop_Recognition_2.image_analysis import get_peak_analysis_of_line
from Drop_Recognition_2.image_analysis import get_peak_detection_image
import cv2
import numpy as np

# Vars
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"
save_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Schriftliche Arbeit/Bilder/Images for section on own Procedure/window_5_analysed.png"
rain_video_frame_self_analysed_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Bilder/rain_video_frame_5_self_analysis.png"
rain_image_2_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Schriftliche Arbeit/Bilder/Images for section on own Procedure/window_5.png"
rain_video_frame_self_analysed = cv2.cvtColor(cv2.imread(rain_video_frame_self_analysed_path), cv2.COLOR_BGR2GRAY)
frame = 0
rain_video = Video(rain_video_path, frame + 1)
rain_image_2 = cv2.cvtColor(cv2.imread(rain_image_2_path), cv2.COLOR_BGR2GRAY)

# Compute Shit
peak_image = get_peak_detection_image(rain_video.get_gray_frame(frame))
peak_image_2 = get_peak_detection_image(rain_image_2)

cv2.imwrite(save_path, cv2.cvtColor(np.float32(peak_image_2) * 255, cv2.COLOR_GRAY2BGR))


