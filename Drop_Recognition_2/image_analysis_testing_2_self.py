from Drop_Recognition_2.video import Video

# Path
rain_video_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Videos/Aufnahmen/rain_video_1.mp4"

# Save Frame 5 for self analysis
rain_video = Video(rain_video_path, 20)
rain_video.save_gray_frame_to_path(5, "C:/Users/Linus/Desktop/Root/Maturarbeit/Bilder/",
                                   "rain_video_frame_5_self_analysis_reference")

