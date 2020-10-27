import matplotlib.pyplot as plt
from Code.Drop_Recognition_2.image_analysis import get_dbscan_clusters_of_line
from Code.Drop_Recognition_2.image_analysis import get_linear_polynominal_coeffs_of_clusters
from Code.Drop_Recognition_2.image_analysis import reduce_list_to_y_value
from Code.Drop_Recognition_2.image_analysis import reduce_list_to_x_value
from Code.Drop_Recognition_2.image_analysis import get_cluster_gaps
from Code.Drop_Recognition_2.image_analysis import absolute_value
from Code.Drop_Recognition_2.image_analysis import get_forward_numerical_first_order_gradient
from Code.Drop_Recognition_2.video import Video
import cv2

# Vars
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"
rain_video = Video(rain_video_path, 30)
y = 200
rain_video_gray_frame_y = rain_video.get_gray_frame(5)[y]


# Nummerical first order differention
dif = []
for i in range(len(rain_video_gray_frame_y) - 2):
    dif.append([i, rain_video_gray_frame_y[i + 1] - rain_video_gray_frame_y[i]])

# Plot
plt.figure(1)
plt.plot(reduce_list_to_y_value(dif))
plt.plot(rain_video_gray_frame_y)


plt.figure(2)
plt.imshow(dif)
plt.show()


