from Code.Drop_Recognition_2.video import Video
import matplotlib.pyplot as plt
import cv2
import pywt

# Vars
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"
rain_video = Video(rain_video_path, 30)

# Wavelet Decomposition
wavelet = pywt.Wavelet("Haar")
rain_video_single_row_wavelet_coefficients = pywt.wavedec(rain_video.get_gray_frame(5)[300], wavelet, level=9)
rain_video_single_row_wavelet_reduced = pywt.waverec(rain_video_single_row_wavelet_coefficients[:-3] + [None] * 3, wavelet)

# Show Data
plt.figure(0)
plt.plot(rain_video_single_row_wavelet_reduced)
plt.plot(rain_video.get_gray_frame(5)[300])
plt.show()
