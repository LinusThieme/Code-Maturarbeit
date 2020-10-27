import matplotlib.pyplot as plt
from Code.Drop_Recognition_2.image_analysis import get_dbscan_clusters_of_line
from Code.Drop_Recognition_2.image_analysis import get_linear_polynominal_coeffs_of_clusters
from Code.Drop_Recognition_2.image_analysis import reduce_list_to_y_value
from Code.Drop_Recognition_2.image_analysis import reduce_list_to_x_value
from Code.Drop_Recognition_2.image_analysis import get_cluster_gaps
from Code.Drop_Recognition_2.image_analysis import absolute_value
from Code.Drop_Recognition_2.image_analysis import get_forward_numerical_first_order_gradient
import statistics
from Code.Drop_Recognition_2.video import Video
import cv2

# Vars
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"
rain_video_frame_self_analysed_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Bilder/rain_video_frame_5_self_analysis.png"
rain_video = Video(rain_video_path, 30)
y = 0
rain_video_gray_frame_y = rain_video.get_gray_frame(5)[y]
rain_video_frame_self_analysed = cv2.cvtColor(cv2.imread(rain_video_frame_self_analysed_path), cv2.COLOR_BGR2GRAY)

# Get Clusters and Cluster Gaps and Cluster linear Coeffs
clusters = get_dbscan_clusters_of_line(rain_video.get_gray_frame(5), y)
gaps = get_cluster_gaps(clusters)
cluster_coeffs = get_linear_polynominal_coeffs_of_clusters(clusters)


# Caclulate D(x) := distance to median of cluster
distance_to_line = []
cluster_id = 0

for x in range(len(rain_video.get_gray_frame(0)[0])):
    if x <= clusters[cluster_id][len(clusters[cluster_id]) - 1][0]:
        linear_value = statistics.median_low(reduce_list_to_y_value(clusters[cluster_id]))
        distance_to_line.append(absolute_value(float(linear_value) - float(rain_video_gray_frame_y[x])))
    else:
        cluster_id += 1

# Calculate gradient of D(x)
gradient = get_forward_numerical_first_order_gradient(distance_to_line)

direct_gradient = get_forward_numerical_first_order_gradient(rain_video_gray_frame_y)

# Plot
plt.figure(0)
for cluster in clusters:
    plt.scatter(x=reduce_list_to_x_value(cluster), y=reduce_list_to_y_value(cluster))


plt.figure(1)
plt.plot(distance_to_line)
plt.plot()

plt.figure(2)
plt.plot(rain_video_gray_frame_y)

plt.figure(3)
plt.plot(distance_to_line)
plt.plot(rain_video_frame_self_analysed[y])


plt.figure(4)
plt.plot(gradient)

plt.figure(5)
plt.plot(gradient, label="Gradient with Filter")
plt.plot(direct_gradient, label="Gradient wihtout Filter")

plt.show()
