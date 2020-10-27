import cv2
from Drop_Recognition_2.image_analysis import get_peak_detection_image

image_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Schriftliche Arbeit/Bilder/Images for section on own Procedure/window_5.png"
save_path = "C:/Users/Linus/Desktop/Root/Maturarbeit/Schriftliche Arbeit/Bilder/Images for section on own Procedure/window_5_overlay.png"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rain_image = get_peak_detection_image(gray_image)

for x in range(0, len(gray_image[0])):
    for y in range(0, len(gray_image)):
        if rain_image[y][x] > 0:
            image[y, x] = [255, 255, 255]


cv2.imwrite(save_path, image)