from Code.Drop_Recognition_2.video import Video
from Code.Drop_Recognition_2.image_analysis import local_average
import matplotlib.pyplot as plt

# Vars
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"
rain_video = Video(rain_video_path, 30)

# Plot
plt.figure(0)
plt.plot(rain_video.get_gray_frame(5)[0], "ro")
plt.plot(local_average(rain_video.get_gray_frame(5)[0], n=7), "bo")
plt.show()


