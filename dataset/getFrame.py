import cv2
import os
import time
from datetime import datetime

url = "http://10.22.169.33:5000/video_feed"

SAVE_FPS = 5  # how many images per second to save

os.makedirs("captured_frames", exist_ok=True)

cap = cv2.VideoCapture(url)

save_interval = 1.0 / SAVE_FPS
last_save = 0

while True:
	ret, frame = cap.read()

	if not ret:
		print("Failed to read frame")
		break

	now = time.time()

	if now - last_save >= save_interval:
		timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]  # trim to milliseconds
		filename = f"captured_frames/{timestamp}.jpg"

		cv2.imwrite(filename, frame)
		print("saved", filename)

		last_save = now

cap.release()