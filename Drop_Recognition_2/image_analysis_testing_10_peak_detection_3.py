import matplotlib.pyplot as plt
from Code.Drop_Recognition_2.image_analysis import absolute_value
from Code.Drop_Recognition_2.image_analysis import get_copy_of_line
from Code.Drop_Recognition_2.image_analysis import multiply_line_by_constant
from Code.Drop_Recognition_2.image_analysis import get_average_function_of_line
from Code.Drop_Recognition_2.image_analysis import get_median_function_of_line
from Code.Drop_Recognition_2.image_analysis import add_constant_to_line
from Code.Drop_Recognition_2.image_analysis import absolute_value_of_line
from Code.Drop_Recognition_2.image_analysis import get_forward_numerical_first_order_gradient
from Code.Drop_Recognition_2.video import Video
import cv2

# Vars
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"
rain_video_frame_self_analysed_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Bilder/rain_video_frame_5_self_analysis.png"
rain_video = Video(rain_video_path, 30)
y = 0
rain_video_gray_frame_y = rain_video.get_gray_frame(5)[y]
rain_video_frame_self_analysed = cv2.cvtColor(cv2.imread(rain_video_frame_self_analysed_path), cv2.COLOR_BGR2GRAY)
average_filter_constant_alpha = 1.8
average_filter_constant_beta = 2
first_gradient_rain_detection_constant = 15

# Get first gradient
rain_video_gray_frame_y_first_gradient = get_forward_numerical_first_order_gradient(rain_video_gray_frame_y)

# Get Second Gradient/ Gradient of Filterd Image
rain_video_gray_frame_y_second_gradient = get_forward_numerical_first_order_gradient(rain_video_gray_frame_y_first_gradient)

# Get Absolute value
rain_video_gray_frame_y_second_gradient_absolute = absolute_value_of_line(rain_video_gray_frame_y_second_gradient)

# Get the Median function of the second gradient absolute line
rain_video_gray_frame_y_second_gradient_filterd_absolute_median = get_median_function_of_line(rain_video_gray_frame_y_second_gradient_absolute, 10)

# Get the average function of the second gradeint absolute line
rain_video_gray_frame_y_second_gradient_filterd_absolute_average_t = multiply_line_by_constant(get_average_function_of_line(rain_video_gray_frame_y_second_gradient_absolute, 30), average_filter_constant_alpha)
rain_video_gray_frame_y_second_gradient_filterd_absolute_average = add_constant_to_line(rain_video_gray_frame_y_second_gradient_filterd_absolute_average_t, average_filter_constant_beta)

# Zero all values below the average value
rain_video_gray_frame_y_second_gradient_absolute_average_filterd = []

for i in range(len(rain_video_gray_frame_y_second_gradient_absolute)):
    if rain_video_gray_frame_y_second_gradient_absolute[i] <= rain_video_gray_frame_y_second_gradient_filterd_absolute_average[i]:
        rain_video_gray_frame_y_second_gradient_absolute_average_filterd.append(0)
    else:
        rain_video_gray_frame_y_second_gradient_absolute_average_filterd.append(rain_video_gray_frame_y_second_gradient_absolute[i])

# Get the Peak values
peak_values = []

for point in rain_video_gray_frame_y_second_gradient_absolute_average_filterd:
    if point > 0:
        peak_values.append(1)
    else:
        peak_values.append(0)

# Get the rain values / do a peak extension via absolute value of first gradient
rain_values = get_copy_of_line(peak_values)

for i in range(len(rain_values)):
    if rain_values[i] == 1:
        # First peak extend left
        j = i - 1
        while j > 0 and not(rain_values[j] == 1) and (absolute_value(rain_video_gray_frame_y_first_gradient[j]) >= first_gradient_rain_detection_constant):
            rain_values[j] = 1
            j -= 1

        # Peak extend Right
        j = i + 1
        while j < (len(rain_values) - 1) and not (rain_values[j] == 1) and (absolute_value(rain_video_gray_frame_y_first_gradient[j]) >= first_gradient_rain_detection_constant):
            rain_values[j] = 1
            j += 1

# Second Phase of Peak extension if the distance between two peaks is one they will be combined

for i in range(len(rain_values) - 3):
    if rain_values[i] == 1 and rain_values[i + 4] == 1:
        rain_values[i + 1] = 1
        rain_values[i + 2] = 1
        rain_values[i + 3] = 1
    elif rain_values[i] == 1 and rain_values[i + 3] == 1:
        rain_values[i + 1] = 1
        rain_values[i + 2] = 1
    elif rain_values[i] == 1 and rain_values[i + 2] == 1:
        rain_values[i + 1] = 1

# Now filter out single peaks

for i in range(1, len(rain_values) - 1):
    if rain_values[i] == 1 and rain_values[i - 1] == 0 and rain_values[i + 1] == 0:
        rain_values[i] = 0

# Plot
figure, axis = plt.subplots(8, 1, constrained_layout=True)
figure.suptitle("Peak Detection")

axis[0].plot(rain_video_gray_frame_y)
axis[0].plot(rain_video_frame_self_analysed[y])
axis[0].set_title("Image Line")

axis[1].plot(rain_video_gray_frame_y_first_gradient)
axis[1].plot(rain_video_frame_self_analysed[y])
axis[1].set_title("First Gradient")

axis[2].plot(rain_video_gray_frame_y_second_gradient)
axis[2].plot(rain_video_frame_self_analysed[y])
axis[2].set_title("Second Gradient")

axis[3].plot(rain_video_gray_frame_y_second_gradient_absolute)
axis[3].plot(rain_video_frame_self_analysed[y])
axis[3].set_title("Second Gradient Absolute")

axis[4].plot(rain_video_gray_frame_y_second_gradient_filterd_absolute_average)
axis[4].plot(rain_video_gray_frame_y_second_gradient_absolute)
axis[4].plot(rain_video_frame_self_analysed[y])
axis[4].set_title("Average Function of Absolute Value of Second Gradient")

axis[5].plot(rain_video_gray_frame_y_second_gradient_absolute_average_filterd)
axis[5].plot(rain_video_frame_self_analysed[y])
axis[5].set_title("Average Filter")

axis[6].plot(peak_values)
axis[6].plot(rain_video_frame_self_analysed[y]/255)
axis[6].set_title("Peak Values")

axis[7].plot(rain_values)
axis[7].plot(rain_video_frame_self_analysed[y]/255)
axis[7].set_title("Rain Values")

plt.show()
