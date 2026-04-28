from ultralytics import YOLO
import cv2
import logging

# Uses YOLO built in track feature. THIS DOES NOT WORK WELL

# =============================
# Load Model
# =============================
model = YOLO("granulatModels/granulat177def.pt")
# Silence ultralytics logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# =============================
# Video Source (HTTP stream)
# =============================
cap = cv2.VideoCapture("test_videos/Best_video.mp4")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
	print("Failed to open video stream")
	exit()

previous_side = {}
cross_count = 0

# =============================
# Main Loop
# =============================
while True:
	ret, frame = cap.read()
	if not ret:
		continue

	frame_height, frame_width = frame.shape[:2]
	border_x = frame_width // 2

	# =============================
	# YOLO Tracking
	# =============================
	results = model.track(
		frame,
		persist=True,
		tracker="bytetrack.yaml",
		conf=0.25,
		verbose=False
	)

	if results[0].boxes is not None and len(results[0].boxes) > 0:

		boxes = results[0].boxes.xyxy.cpu().numpy()

		if results[0].boxes.id is not None:
			ids = results[0].boxes.id.cpu().numpy()
		else:
			ids = range(len(boxes))

		for box, obj_id in zip(boxes, ids):

			x1, y1, x2, y2 = box
			obj_id = int(obj_id)

			cv2.rectangle(
				frame,
				(int(x1), int(y1)),
				(int(x2), int(y2)),
				(255, 0, 0),
				2
			)

			# =============================
			# Determine FULL box side
			# =============================
			if x2 < border_x:
				current_side = "left"
			elif x1 > border_x:
				current_side = "right"
			else:
				current_side = "crossing"

			# =============================
			# Crossing Logic
			# =============================
			if obj_id not in previous_side:

				if current_side in ["left", "right"]:
					previous_side[obj_id] = current_side

			else:
				prev_side = previous_side[obj_id]

				if prev_side == "left" and current_side == "right":
					cross_count += 1
					previous_side[obj_id] = "right"

				elif prev_side == "right" and current_side == "left":
					cross_count += 1
					previous_side[obj_id] = "left"

				elif current_side in ["left", "right"]:
					previous_side[obj_id] = current_side

	# =============================
	# Draw border line
	# =============================
	cv2.line(
		frame,
		(border_x, 0),
		(border_x, frame_height),
		(0, 255, 0),
		2
	)

	# =============================
	# Draw counter
	# =============================
	cv2.putText(
		frame,
		f"Cross Count: {cross_count}",
		(50, 50),
		cv2.FONT_HERSHEY_SIMPLEX,
		2,
		(255, 0, 0),
		2
	)

	cv2.imshow("Whole Box Crossing Counter", frame)

	if cv2.waitKey(1) & 0xFF == 27:
		break

cap.release()
cv2.destroyAllWindows()