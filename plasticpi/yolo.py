from ultralytics import YOLO
from picamera2 import Picamera2
import time

# Load your model (keep it small if possible!)
model = YOLO("model/train.pt")

# Initialize camera
picam2 = Picamera2()

video_config = picam2.create_video_configuration(
	main={"size": (640, 480)},
	lores={"size": (224, 224), "format": "RGB888"},  # small frame for YOLO
	controls={"FrameRate": 10}  # lower FPS to reduce load
)

picam2.configure(video_config)
picam2.start()
time.sleep(0.5)
picam2.set_controls({
	"AfMode": 0,
	"LensPosition": 10
})

frame_counter = 0

try:
	while True:
		frame = picam2.capture_array("lores")
		frame_counter += 1

		# Run YOLO only every N frames to reduce load
		if frame_counter % 30 == 0:
			start = time.time()
			results = model(frame)
			print("Inference time:", round(time.time() - start, 2), "s")

			if results[0].boxes is not None:
				print("Detections:", len(results[0].boxes))

except KeyboardInterrupt:
	print("Exiting...")

finally:
	picam2.stop()
