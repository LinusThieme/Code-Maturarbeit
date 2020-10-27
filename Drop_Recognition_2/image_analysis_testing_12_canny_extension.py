import cv2
import matplotlib.pyplot as plt
import numpy as np
from Code.Drop_Recognition_2.video import Video
from Code.Drop_Recognition_2.image_analysis import get_peak_analysis_of_line
from Code.Drop_Recognition_2.image_analysis import get_copy_of_line
from Code.Drop_Recognition_2.image_analysis import absolute_value

# Vars
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"
rain_video_frame_self_analysed_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Bilder/rain_video_frame_5_self_analysis.png"
rain_video = Video(rain_video_path, 30)
y = 0
frame = 5
rain_video_gray_frame_y = rain_video.get_gray_frame(frame)[y]
rain_video_frame_self_analysed = cv2.cvtColor(cv2.imread(rain_video_frame_self_analysed_path), cv2.COLOR_BGR2GRAY)
rain_detection_minimal_first_gradient = 15
np_line_first_gradient = np.delete(np.convolve(rain_video_gray_frame_y, np.array([1, -1])), 0)

# Get Peak analysis
peak_analysis_values = get_peak_analysis_of_line(rain_video_gray_frame_y)

# Get Canny Image
canny_rain_image = cv2.Canny(rain_video.get_gray_frame(frame), 10, 120, apertureSize=3, L2gradient=True)

# Extend the Peaks of the Canny Image
rain_values = np.array(get_copy_of_line(canny_rain_image[y]))/255
for i in range(len(rain_values)):
    if rain_values[i] == 1:
        # Peak extension left
        j = i - 1
        while j > 0 and not (rain_values[j] == 1) and (absolute_value(np_line_first_gradient[j] >= rain_detection_minimal_first_gradient)):
            rain_values[j] = 1
            j -= 1
        # Peak extension right
        j = i + 1
        while j < (len(rain_values)) and not (rain_values[j] == 1) and (absolute_value(np_line_first_gradient[j]) > rain_detection_minimal_first_gradient):
            rain_values[j] = 1
            j += 1


# Plot
figure, axis = plt.subplots(4, 1, constrained_layout=True)
figure.suptitle("Peak Detection via Canny and Peak Analysis")

axis[0].plot(rain_video_gray_frame_y)
axis[0].plot(rain_video_frame_self_analysed[y])
axis[0].set_title("Image Line")

axis[1].plot(rain_values)
axis[1].plot(rain_video_frame_self_analysed[y]/255)
axis[1].set_title("Canny Values with peak extension")

axis[2].plot(peak_analysis_values)
axis[2].plot(rain_video_frame_self_analysed[y]/255)
axis[2].set_title("Peak Detection Values")

axis[3].plot(peak_analysis_values)
axis[3].plot(rain_values)
axis[3].set_title("Peak Detection Values and Canny Values Comparisson")

plt.show()
