from Code.Drop_Recognition_2.video import Video
from Code.Drop_Recognition_2.image_analysis import reduce_list_to_y_value
from Code.Drop_Recognition_2.image_analysis import get_linear_polynominal_coeffs
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

# Vars
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"
rain_video = Video(rain_video_path, 30)
frame = 5
y = 700

# Create Dataset for DBSCAN
rain_video_line_zero_points = []
for i in range(len(rain_video.get_gray_frame(frame)[y])):
    rain_video_line_zero_points.append([i, rain_video.get_gray_frame(frame)[y][i]])

# DBSCAN
cluster_analysis = DBSCAN(eps=25, min_samples=25).fit_predict(rain_video_line_zero_points)

# Cluster Sorting
clusters = []
for i in range(max(cluster_analysis) + 1):
    clusters.append([])
    for j in range(len(rain_video_line_zero_points)):
        if cluster_analysis[j] == i:
            clusters[i].append(rain_video_line_zero_points[j])

# Linear Regression of Clusters
linear_regression_coeffs = []
for i in range(len(clusters)):
    linear_regression_coeffs.append(get_linear_polynominal_coeffs(clusters[i]))

# Show results
plt.figure(0)
plt.plot(reduce_list_to_y_value(clusters[1]), "bo")
xx = np.linspace(0, 500, 2500)
yy = linear_regression_coeffs[1][0] + linear_regression_coeffs[1][1]*xx
plt.plot(xx, yy, label="Regression: ax + b")


plt.figure(1)
plt.plot(rain_video.get_gray_frame(frame)[y])

plt.show()
