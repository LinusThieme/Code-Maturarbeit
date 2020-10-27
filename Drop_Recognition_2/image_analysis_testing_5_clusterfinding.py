from Code.Drop_Recognition_2.video import Video
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


# Vars
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"
rain_video = Video(rain_video_path, 30)
y = 700

# Create Dataset
rain_video_line_zero_points = []
for i in range(len(rain_video.get_gray_frame(5)[y])):
    rain_video_line_zero_points.append([i, rain_video.get_gray_frame(5)[y][i]])


# Clusterfinding via DBSCAN
clustering = DBSCAN(eps=25, min_samples=25).fit_predict(rain_video_line_zero_points)


# Seperate the Clusters
clusters = []
for i in range(len(rain_video.get_gray_frame(5)[y])):
    if clustering[i] >= 0:
        clusters.append(rain_video.get_gray_frame(5)[y][i])
    else:
        clusters.append(None)


# Show results
plt.figure(5)
plt.plot(rain_video.get_gray_frame(5)[y])

plt.figure(0)
plt.plot(rain_video.get_gray_frame(5)[y], "bo")

plt.figure(1)
plt.plot(clustering, "bo")

plt.figure(2)
plt.plot(clusters, "ro")

plt.figure(3)
plt.plot(rain_video.get_gray_frame(5)[y], "ro")
plt.plot(clusters, "bo")

plt.show()
