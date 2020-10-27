import cv2
import numpy as np
from Drop_Recognition_2.video import Video
from Drop_Recognition_2.image_analysis import get_average_function_of_line
from Drop_Recognition_2.image_analysis import absolute_value
from Drop_Recognition_2.image_analysis import get_copy_of_line


import matplotlib.pyplot as plt

# Compare the following analysis forms of line 0 of frame 5: Canny, (Peak) at the moment just line 0, Self.
# This is done to determine the effectivnes of the combined frequenzy/peak + canny analysis.

# vars
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"
rain_video_frame_self_analysed_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Bilder/rain_video_frame_5_self_analysis.png"
rain_video = Video(rain_video_path, 20)
rain_video_frame_self_analysed = cv2.cvtColor(cv2.imread(rain_video_frame_self_analysed_path), cv2.COLOR_BGR2GRAY)
rain_video_frame_canny = cv2.Canny(rain_video.get_gray_frame(5), 10, 120, apertureSize=3, L2gradient=True)
average_filter_range = 30
linear_transform_alpha = 1.8
linear_transform_beta = 2
rain_detection_minimal_first_gradient = 15

# Create shit to show it in figures an stuff
line = rain_video.get_gray_frame(5)[0]
np_line = np.array(line)

# Frist Gradient
np_line_first_gradient = np.delete(np.convolve(np_line, np.array([1, -1])), 0)

# Second Gradient
np_line_second_gradient = np.delete(np.convolve(np_line_first_gradient, np.array([1, -1])), 0)

# Absolute Value
np_line_second_gradient_abs = np.abs(np_line_second_gradient)

# Local Average Filtering
np_line_second_gradient_abs_avg_filterd = np.array(get_average_function_of_line(np_line_second_gradient_abs.tolist(), average_filter_range))

# Linear transformation of line
np_line_second_gradient_abs_avg_filterd_transformed = np_line_second_gradient_abs_avg_filterd * linear_transform_alpha + linear_transform_beta

# Get Peak Values
peak_values = []
for i in range(len(np_line_second_gradient_abs)):
    if np_line_second_gradient_abs[i] <= np_line_second_gradient_abs_avg_filterd_transformed[i]:
        peak_values.append(0)
    else:
        peak_values.append(1)

rain_values = get_copy_of_line(peak_values)
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

# Peak combination
for i in range(len(rain_values) - 4):
    if rain_values[i] == 1 and rain_values[i + 4] == 1:
        rain_values[i + 1] = 1
        rain_values[i + 2] = 1
        rain_values[i + 3] = 1
    elif rain_values[i] == 1 and rain_values[i + 3] == 1:
        rain_values[i + 1] = 1
        rain_values[i + 2] = 1
    elif rain_values[i] == 1 and rain_values[i + 2] == 1:
        rain_values[i + 1] = 1
# Filter out single peaks
for i in range(1, len(rain_values) - 1):
    if rain_values[i] == 1 and rain_values[i - 1] == 0 and rain_values[i + 1] == 0:
        rain_values[i] = 0


for i in range(len(rain_values)):
    if rain_values[i] == 1:
        rain_values[i] = 255

# Show Data
plt.figure(0)

plt.plot(rain_video_frame_self_analysed[0])

plt.xlabel("X Koordinate des Pixels der Reihe 0")
plt.ylabel("Intensität")
plt.show()
