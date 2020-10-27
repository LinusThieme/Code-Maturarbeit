import numpy as np
import cv2
import matplotlib.pyplot as plt
from Drop_Recognition_2.video import Video
from Drop_Recognition_2.image_analysis import add_lines_to_image, remove_zero

# vars
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"
rain_video_frame_self_analysed_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Bilder/rain_video_frame_5_self_analysis.png"
rain_video = Video(rain_video_path, 20)
rain_video_frame_self_analysed = cv2.cvtColor(cv2.imread(rain_video_frame_self_analysed_path), cv2.COLOR_BGR2GRAY)

rain_video.save_video_to_path("C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Created/", "test_save")


# Show a frame and a grayscale frame to test
plt.figure(2)
plt.set_cmap("gray")
plt.imshow(rain_video.get_gray_frame(5))


# Canny Filter
canny_rain_image = cv2.Canny(rain_video.get_gray_frame(5), 10, 120, apertureSize=3, L2gradient=True)
cv2.imwrite("C:/Users/Linus/Desktop/Root/Maturarbeit/Bilder/original_image.png", rain_video.get_gray_frame(5))
cv2.imwrite("C:/Users/Linus/Desktop/Root/Maturarbeit/Bilder/canny_rain_image.png", canny_rain_image)
cv2.imwrite("C:/Users/Linus/Desktop/Root/Maturarbeit/Bilder/canny_rain_image_with_background.png", add_lines_to_image(rain_video.get_gray_frame(5), canny_rain_image))

# Add Lines to Grayscale Image
plt.figure(4)
plt.imshow(add_lines_to_image(rain_video.get_gray_frame(5), canny_rain_image))

# Compare canny to self analysed
plt.figure(5)
plt.plot(canny_rain_image[0])
plt.plot(rain_video_frame_self_analysed[0])
plt.legend()

# Show the figures
plt.show()
