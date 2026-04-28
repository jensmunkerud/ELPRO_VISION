from ultralytics import YOLO
import cv2

# Uses basic, custom track feature. THIS DOES NOT WORK WELL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "granulatModels/granulat177def.pt"
VIDEO_PATH = "test_videos/lottaGranulat.mp4"

ZONE_HALF_W = 0.08   # zone = center ± 8% of frame width

LINE_COLOR = (0, 255, 0)
ZONE_COLOR = (0, 180, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)
# ---------------------------------------------------------------------------

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
	raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

LINE_X = frame_width // 2
ZONE_L = int(LINE_X - ZONE_HALF_W * frame_width)
ZONE_R = int(LINE_X + ZONE_HALF_W * frame_width)

print(f"Frame: {frame_width}x{frame_height}  LINE_X={LINE_X}  ZONE=[{ZONE_L},{ZONE_R}]")

# ---------------------------------------------------------------------------
# Crossing tracking
# Each detected box keeps a short history of its center positions
# ---------------------------------------------------------------------------
cx_histories = []
total_crossings = 0
MAX_HISTORY = 10  # number of frames to remember for each box


def get_side(cx):
	if cx < ZONE_L:
		return "left"
	if cx > ZONE_R:
		return "right"
	return "zone"


def draw_overlay(frame, crossings, n_boxes):
	overlay = frame.copy()
	cv2.rectangle(overlay, (ZONE_L, 0), (ZONE_R, frame_height), (0, 60, 0), -1)
	cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

	cv2.line(frame, (ZONE_L, 0), (ZONE_L, frame_height), ZONE_COLOR, 2)
	cv2.line(frame, (ZONE_R, 0), (ZONE_R, frame_height), ZONE_COLOR, 2)
	cv2.line(frame, (LINE_X, 0), (LINE_X, frame_height), LINE_COLOR, 2)

	lines = [f"Crossings: {crossings}", f"In frame:  {n_boxes}"]
	pad = 8
	lh = 24
	cv2.rectangle(frame, (0, 0), (220, pad * 2 + lh * len(lines)), TEXT_BG, -1)
	for i, text in enumerate(lines):
		cv2.putText(frame, text, (pad, pad + lh * i + 16),
					cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 1, cv2.LINE_AA)


while True:
	ret, frame = cap.read()
	if not ret:
		break

	results = model(frame, conf=0.2)[0]  # just detect, no tracker
	boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
	n_boxes = len(boxes)

	current_cx = [(box[0] + box[2]) / 2.0 for box in boxes]

	# Update histories
	new_histories = []
	for cx in current_cx:
		matched = False
		for hist in cx_histories:
			# match if last cx is close (allow some pixel tolerance)
			if abs(cx - hist[-1]) < 50:
				hist.append(cx)
				if len(hist) > MAX_HISTORY:
					hist.pop(0)
				new_histories.append(hist)
				matched = True
				break
		if not matched:
			new_histories.append([cx])
	cx_histories = new_histories

	# Count crossings: check last history of each object
	for hist in cx_histories:
		if len(hist) < 2:
			continue
		if get_side(hist[0]) == "right" and get_side(hist[-1]) == "left":
			if any(ZONE_L <= c <= ZONE_R for c in hist):
				total_crossings += 1
				hist.clear()  # prevent double-counting

	draw_overlay(frame, total_crossings, n_boxes)
	cv2.imshow("Granulat Counter", frame)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
print(f"Done. Total crossings: {total_crossings}")