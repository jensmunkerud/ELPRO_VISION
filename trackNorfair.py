from ultralytics import YOLO
import torch
from norfair import Detection, Tracker
import cv2
import numpy as np

# Simple settings
modelPath = "granulatModels/granulat177def.pt"
source = "test_videos/Best_Video.mp4"
saveVideo = False
outputPath = "results/Best_Video_output.mp4"
outputFps = 0.0

model = YOLO(modelPath)

# Auto-select device: prefer CUDA, then Apple MPS, else CPU
if torch.cuda.is_available():
	device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"

# Try to move model to chosen device (some ultralytics versions handle this internally)
try:
	model.to(device)
except Exception:
	pass

# Norfair tracker setup
tracker = Tracker(
	distance_function="euclidean",
	distance_threshold=50,
	hit_counter_max=200,
	initialization_delay=1
)

cap = cv2.VideoCapture(source)
if not cap.isOpened():
	raise RuntimeError(f"Could not open video source: {source}")

writer = None
if saveVideo:
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
	source_fps = cap.get(cv2.CAP_PROP_FPS)
	fps = outputFps if outputFps > 0 else (source_fps if source_fps and source_fps > 0 else 30.0)
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))
	if not writer.isOpened():
		raise RuntimeError(f"Could not open output file for writing: {outputPath}")

# Counting state
count_line_x = None
last_side_by_id = {}
counted_ids = set()
total_crossed = 0

# CONFIDENCE THRESHOLD
CONF_THRES = 0.05

while True:
	ok, frame = cap.read()
	if not ok:
		break

	# Draw vertical count line
	h, w = frame.shape[:2]
	if count_line_x is None:
		count_line_x = w // 2  # vertical count line in the middle
	cv2.line(frame, (count_line_x, 0), (count_line_x, h), (0, 255, 255), 2)

	# YOLO inference on current frame
	result = model(frame, device=device, verbose=True)[0]

	# Convert YOLO detections to Norfair detections
	detections = []
	if result.boxes is not None and len(result.boxes) > 0:
		for box in result.boxes:
			conf = float(box.conf[0])
			if conf < CONF_THRES:
				continue

			x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
			cx = (x1 + x2) / 2.0
			cy = (y1 + y2) / 2.0

			detections.append(
				Detection(
					points=np.array([[cx, cy]], dtype=np.float32),
					scores=np.array([conf], dtype=np.float32),
					data={"bbox": (int(x1), int(y1), int(x2), int(y2))}
				)
			)

	# Draw raw YOLO boxes
	for det in detections:
		x1, y1, x2, y2 = det.data["bbox"]
		cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 1)
		conf_val = float(det.scores[0])
		cv2.putText(
			frame, f"{conf_val:.2f}", (x1, y1 - 4),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2, cv2.LINE_AA
		)

	tracked_objects = tracker.update(detections=detections)

	# Draw tracked objects + count crossings
	for obj in tracked_objects:
		x, y = obj.estimate[0]  # one-point tracking (centroid)
		track_id = obj.id

		# Draw point + id
		cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
		cv2.putText(
			frame, f"ID:{track_id}", (int(x) + 6, int(y) - 6),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
		)

		# Side of line: -1 left, +1 right
		current_side = -1 if x < count_line_x else 1
		previous_side = last_side_by_id.get(track_id)

		# Count each track once when it crosses line
		if previous_side is not None and previous_side != current_side and track_id not in counted_ids:
			total_crossed += 1
			counted_ids.add(track_id)
			print(f"TOTAL CROSSED: {total_crossed}")

		last_side_by_id[track_id] = current_side

	# Overlay count
	cv2.putText(
		frame, f"Total : {total_crossed}", (w//2 - 180, h//2),
		cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
	)

	if writer is not None:
		writer.write(frame)

	cv2.imshow("Granulate Tracking + Counting (Norfair)", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
if writer is not None:
	writer.release()
cv2.destroyAllWindows()