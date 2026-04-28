from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("granulatModels/granulat177def.pt")

# --- State ---
total_crossings = 0
prev_left_count = 0
prev_right_count = 0

def get_centers(results):
	"""Extract (cx, cy) for every detected box in a result frame."""
	centers = []
	if results[0].boxes is not None:
		for box in results[0].boxes.xyxy.cpu().numpy():
			x1, y1, x2, y2 = box
			cx = (x1 + x2) / 2
			cy = (y1 + y2) / 2
			centers.append((cx, cy))
	return centers


cap = cv2.VideoCapture("test_videos/littleGranulat.mp4")
# cap = cv2.VideoCapture("http://10.22.168.232:5000/video_feed")
# cap = cv2.VideoCapture(1)

if not cap.isOpened():
	print("Error: could not open video source.")
	exit()

frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = frame_width // 2

# Run model tracking frame-by-frame so we can draw on top
while True:
	ret, frame = cap.read()
	if not ret:
		break

	# ── YOLO inference (persist=True keeps tracker state across frames) ──
	results = model(frame, verbose=False)

	# ── Annotated frame from YOLO (boxes + IDs already drawn) ──
	annotated = results[0].plot()

	# ── Compute centers & region counts ──
	centers = get_centers(results)
	left_count  = sum(1 for (cx, _) in centers if cx <= center_x)
	right_count = sum(1 for (cx, _) in centers if cx >  center_x)

	# ── Crossing logic:
	#    A granule crossed LEFT→RIGHT when right drops by 1 AND left rises by 1
	delta_right = right_count - prev_right_count   # negative = fewer on right
	delta_left  = left_count  - prev_left_count    # positive = more on left

	if delta_right == -1 and delta_left == 1:
		total_crossings += 1

	prev_left_count  = left_count
	prev_right_count = right_count

	# ── Draw vertical center line ──
	cv2.line(annotated, (center_x, 0), (center_x, frame_height), (0, 255, 255), 2)

	# ── Draw center dots on each detected object ──
	for (cx, cy) in centers:
		color = (255, 100, 0) if cx <= center_x else (0, 100, 255)
		cv2.circle(annotated, (int(cx), int(cy)), 5, color, -1)

	# ── Region labels ──
	cv2.putText(annotated, f"LEFT: {left_count}",
				(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)
	cv2.putText(annotated, f"RIGHT: {right_count}",
				(center_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)

	# ── Total crossings counter (bottom centre) ──
	label = f"Total crossings (R->L): {total_crossings}"
	(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
	tx = (frame_width - tw) // 2
	cv2.rectangle(annotated, (tx - 6, frame_height - 45),
				  (tx + tw + 6, frame_height - 10), (0, 0, 0), -1)
	cv2.putText(annotated, label,
				(tx, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

	cv2.imshow("Granulat Tracker", annotated)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal total crossings (right→left): {total_crossings}")