from ultralytics import YOLO
import torch
from norfair import Detection, Tracker
import cv2
import numpy as np

modelPath = "granulatModels/multidetect.pt"
source = "test_videos/hispeed.mp4"
saveVideo = False
outputPath = "results/MULTITRACK.mp4"
outputFps = 0.0

model = YOLO(modelPath)

if torch.cuda.is_available():
	device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"

try:
	model.to(device)
except Exception:
	pass

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

# Counting state — now tracks top/bottom crossing
count_line_y = None
last_side_by_id = {}
counted_ids = set()
crossed_by_class = {"Granulat": 0, "Perle": 0}

CLASS_NAMES = model.names
track_class_by_id = {}
CONF_THRES = 0.05
CLASS_COLORS = {"Granulat": (255, 100, 0), "Perle": (180, 0, 255)}

while True:
	ok, frame = cap.read()
	if not ok:
		break

	h, w = frame.shape[:2]
	if count_line_y is None:
		count_line_y = h // 2

	# Horizontal count line
	cv2.line(frame, (0, count_line_y), (w, count_line_y), (0, 255, 255), 2)

	result = model(frame, device=device, verbose=False)[0]

	detections = []
	if result.boxes is not None and len(result.boxes) > 0:
		for box in result.boxes:
			conf = float(box.conf[0])
			if conf < CONF_THRES:
				continue
			x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
			cx = (x1 + x2) / 2.0
			cy = (y1 + y2) / 2.0
			cls_idx = int(box.cls[0])
			cls_name = CLASS_NAMES.get(cls_idx, str(cls_idx))
			detections.append(
				Detection(
					points=np.array([[cx, cy]], dtype=np.float32),
					scores=np.array([conf], dtype=np.float32),
					data={"bbox": (int(x1), int(y1), int(x2), int(y2)), "cls_name": cls_name}
				)
			)

	for det in detections:
		x1, y1, x2, y2 = det.data["bbox"]
		cls_name = det.data["cls_name"]
		color = CLASS_COLORS.get(cls_name, (200, 200, 200))
		cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
		cv2.putText(frame, f"{cls_name} {float(det.scores[0]):.2f}", (x1, y1 - 4),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

	tracked_objects = tracker.update(detections=detections)
	det_points = [(d.points[0], d.data["cls_name"]) for d in detections]

	for obj in tracked_objects:
		x, y = obj.estimate[0]
		track_id = obj.id

		if det_points:
			dists = [np.linalg.norm(np.array([x, y]) - np.array(pt)) for pt, _ in det_points]
			nearest_idx = int(np.argmin(dists))
			if dists[nearest_idx] < 50:
				track_class_by_id[track_id] = det_points[nearest_idx][1]

		cls_name = track_class_by_id.get(track_id, "Unknown")
		dot_color = CLASS_COLORS.get(cls_name, (0, 255, 0))

		cv2.circle(frame, (int(x), int(y)), 4, dot_color, -1)
		cv2.putText(frame, f"ID:{track_id} {cls_name}", (int(x) + 6, int(y) - 6),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, dot_color, 1, cv2.LINE_AA)

		# Top/bottom side using y coordinate
		current_side = -1 if y < count_line_y else 1
		previous_side = last_side_by_id.get(track_id)

		if previous_side is not None and previous_side != current_side and track_id not in counted_ids:
			if cls_name in crossed_by_class:
				crossed_by_class[cls_name] += 1
			counted_ids.add(track_id)
			print(f"Crossed: {cls_name} | Granulat: {crossed_by_class['Granulat']}  Perle: {crossed_by_class['Perle']}")

		last_side_by_id[track_id] = current_side

	cv2.putText(frame, f"Granulat: {crossed_by_class['Granulat']}", (20, 50),
				cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 0), 2, cv2.LINE_AA)
	cv2.putText(frame, f"Perle: {crossed_by_class['Perle']}", (20, 100),
				cv2.FONT_HERSHEY_SIMPLEX, 1.5, (180, 0, 255), 2, cv2.LINE_AA)

	if writer is not None:
		writer.write(frame)

	cv2.imshow("Granulate Tracking + Counting (Norfair)", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
if writer is not None:
	writer.release()
cv2.destroyAllWindows()

print(f"\nFinal count — Granulat: {crossed_by_class['Granulat']}, Perle: {crossed_by_class['Perle']}")