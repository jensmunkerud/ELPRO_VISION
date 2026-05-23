import argparse
import importlib.metadata
import os
import threading
import time
from flask import Flask, request, Response, redirect, url_for, render_template_string, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.after_request
def add_cors_headers(resp):
	resp.headers["Access-Control-Allow-Origin"] = "*"
	resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
	resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
	return resp


@app.route("/analyze", methods=["OPTIONS"])
@app.route("/analysis_status", methods=["OPTIONS"])
@app.route("/analysis_feed", methods=["OPTIONS"])
@app.route("/cancel_analysis", methods=["OPTIONS"])
def cors_preflight():
	return ("", 204)

DEFAULT_MODEL_PATH = "granulatModels/granulat177def.pt"
DEFAULT_SOURCE = "test_videos/video.mp4"
CONF_THRES = 0.05
OUTPUT_STREAM_HEIGHT = 1080
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
MODEL_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "models")
ALLOWED_VIDEO_EXTENSIONS = {".mp4"}
ALLOWED_MODEL_EXTENSIONS = {".pt"}

analysis_condition = threading.Condition()
analysis_running = False
analysis_status = "idle"
analysis_error = ""
analysis_frame = None
analysis_frame_seq = 0
analysis_total = 0
analysis_job_id = ""
analysis_video_file = ""
analysis_model_path = ""
analysis_cancel_event = None
ORIGINAL_IMPORTLIB_METADATA_VERSION = importlib.metadata.version


def _import_norfair():
	def version_with_fallback(distribution_name):
		if distribution_name == "norfair":
			try:
				return ORIGINAL_IMPORTLIB_METADATA_VERSION(distribution_name)
			except importlib.metadata.PackageNotFoundError:
				return "0.0.0"
		return ORIGINAL_IMPORTLIB_METADATA_VERSION(distribution_name)

	importlib.metadata.version = version_with_fallback
	from norfair import Detection, Tracker
	return Detection, Tracker


def iter_annotated_frames(source, model_path=DEFAULT_MODEL_PATH, conf_thres=CONF_THRES, should_stop=None):
	from ultralytics import YOLO
	import cv2
	import numpy as np

	Detection, Tracker = _import_norfair()

	model = YOLO(model_path)

	tracker = Tracker(
		distance_function="euclidean",
		distance_threshold=50,
		hit_counter_max=200,
		initialization_delay=1
	)

	cap = cv2.VideoCapture(source)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video source: {source}")

	count_line_x = None
	last_side_by_id = {}
	counted_ids = set()
	total_crossed = 0

	try:
		while True:
			if should_stop is not None and should_stop():
				break

			ok, frame = cap.read()
			if not ok:
				break

			h, w = frame.shape[:2]
			if count_line_x is None:
				count_line_x = w // 2

			cv2.line(frame, (count_line_x, 0), (count_line_x, h), (0, 255, 255), 2)

			if should_stop is not None and should_stop():
				break

			result = model(frame, verbose=False)[0]

			detections = []
			if result.boxes is not None:
				for box in result.boxes:
					conf = float(box.conf[0])
					if conf < conf_thres:
						continue

					x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
					cx = (x1 + x2) / 2
					cy = (y1 + y2) / 2

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


			tracked = tracker.update(detections=detections)

			for obj in tracked:
				x, y = obj.estimate[0]
				tid = obj.id

				# Draw point + id
				cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
				cv2.putText(
					frame, f"ID:{tid}", (int(x) + 6, int(y) - 6),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
				)

				current_side = -1 if x < count_line_x else 1
				prev = last_side_by_id.get(tid)

				if prev is not None and prev != current_side and tid not in counted_ids:
					total_crossed += 1
					counted_ids.add(tid)
					print("TOTAL:", total_crossed)

				last_side_by_id[tid] = current_side

			cv2.putText(
				frame,
				f"Total: {total_crossed}",
				(50, 50),
				cv2.FONT_HERSHEY_SIMPLEX,
				1,
				(0, 0, 255),
				2,
			)

			yield frame, total_crossed
	finally:
		cap.release()


def analyze_video_stream(source, model_path=DEFAULT_MODEL_PATH, conf_thres=CONF_THRES, jpeg_quality=80, should_stop=None):
	import cv2

	encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
	for frame, total_crossed in iter_annotated_frames(source, model_path=model_path, conf_thres=conf_thres, should_stop=should_stop):
		if should_stop is not None and should_stop():
			break
		h, w = frame.shape[:2]
		if h > 0 and h != OUTPUT_STREAM_HEIGHT:
			scaled_w = max(1, int(round((w * OUTPUT_STREAM_HEIGHT) / h)))
			frame = cv2.resize(frame, (scaled_w, OUTPUT_STREAM_HEIGHT), interpolation=cv2.INTER_AREA)
		ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
		if not ok:
			continue
		yield {"jpeg": jpeg.tobytes(), "total": total_crossed}


def list_available_models():
	model_paths = []
	search_dirs = [
		os.path.join(os.path.dirname(__file__), "granulatModels"),
		os.path.join(os.path.dirname(__file__), "weights"),
	]

	for root_dir in search_dirs:
		if not os.path.isdir(root_dir):
			continue
		for root, _, files in os.walk(root_dir):
			for name in files:
				if name.lower().endswith(".pt"):
					full_path = os.path.join(root, name)
					rel_path = os.path.relpath(full_path, os.path.dirname(__file__))
					model_paths.append(rel_path)

	if DEFAULT_MODEL_PATH not in model_paths:
		model_paths.append(DEFAULT_MODEL_PATH)

	model_paths = sorted(set(model_paths))
	return model_paths


def _is_allowed_video(filename):
	_, ext = os.path.splitext(filename or "")
	return ext.lower() in ALLOWED_VIDEO_EXTENSIONS


def _is_allowed_model(filename):
	_, ext = os.path.splitext(filename or "")
	return ext.lower() in ALLOWED_MODEL_EXTENSIONS


def _resolve_model_path(model_path):
	candidate = (model_path or "").strip()
	if not candidate:
		candidate = DEFAULT_MODEL_PATH

	if os.path.isabs(candidate):
		return candidate

	return os.path.join(os.path.dirname(__file__), candidate)


def _analysis_worker(input_path, model_path, job_id, cancel_event):
	global analysis_running, analysis_status, analysis_error
	global analysis_frame, analysis_frame_seq, analysis_total

	try:
		for update in analyze_video_stream(input_path, model_path=model_path, should_stop=cancel_event.is_set):
			with analysis_condition:
				if analysis_job_id != job_id:
					return
				if cancel_event.is_set():
					analysis_running = False
					analysis_status = "cancelled"
					analysis_condition.notify_all()
					return
				analysis_frame = update.get("jpeg")
				analysis_total = int(update.get("total", analysis_total))
				analysis_frame_seq += 1
				analysis_condition.notify_all()

		with analysis_condition:
			if analysis_job_id == job_id:
				analysis_running = False
				if cancel_event.is_set():
					analysis_status = "cancelled"
				else:
					analysis_status = "completed"
				analysis_condition.notify_all()
	except Exception as err:
		with analysis_condition:
			if analysis_job_id == job_id:
				analysis_running = False
				if cancel_event.is_set():
					analysis_status = "cancelled"
					analysis_error = ""
				else:
					analysis_status = "error"
					analysis_error = str(err)
				analysis_condition.notify_all()


def start_analysis_job(input_path, original_filename, model_path):
	global analysis_running, analysis_status, analysis_error
	global analysis_frame, analysis_frame_seq, analysis_total
	global analysis_job_id, analysis_video_file, analysis_model_path, analysis_cancel_event

	with analysis_condition:
		if analysis_running:
			return False, "Analysis already running"

		cancel_event = threading.Event()
		analysis_job_id = str(int(time.time() * 1000))
		analysis_video_file = original_filename
		analysis_model_path = model_path
		analysis_cancel_event = cancel_event
		analysis_running = True
		analysis_status = "running"
		analysis_error = ""
		analysis_frame = None
		analysis_frame_seq = 0
		analysis_total = 0

	threading.Thread(target=_analysis_worker, args=(input_path, model_path, analysis_job_id, cancel_event), daemon=True).start()
	return True, "Analysis started"


def cancel_analysis_job():
	global analysis_running, analysis_status, analysis_error, analysis_cancel_event

	with analysis_condition:
		if not analysis_running:
			return False, "No running analysis"

		if analysis_cancel_event is not None:
			analysis_cancel_event.set()

		analysis_running = False
		analysis_status = "cancelled"
		analysis_error = ""
		analysis_condition.notify_all()
		return True, "Cancel requested"


def generate_analysis_frames(job_id):
	last_seq = -1
	while True:
		with analysis_condition:
			while True:
				if not job_id or analysis_job_id != job_id:
					return
				if analysis_frame is not None and analysis_frame_seq != last_seq:
					frame = analysis_frame
					last_seq = analysis_frame_seq
					break
				if not analysis_running:
					return
				analysis_condition.wait(timeout=1.0)

		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def run_analysis(source=DEFAULT_SOURCE, model_path=DEFAULT_MODEL_PATH):
	import cv2

	for frame, _ in iter_annotated_frames(source, model_path=model_path):
		cv2.imshow("Analysis", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	cv2.destroyAllWindows()


@app.route("/run_analysis", methods=["POST"])
def trigger():
	source = request.form.get("source", DEFAULT_SOURCE)
	model_path = _resolve_model_path(request.form.get("model_path", DEFAULT_MODEL_PATH))
	threading.Thread(target=run_analysis, args=(source, model_path), daemon=True).start()
	return "running"


@app.route("/")
def index():
	os.makedirs(UPLOAD_DIR, exist_ok=True)
	os.makedirs(MODEL_UPLOAD_DIR, exist_ok=True)
	message = request.args.get("message", "")
	with analysis_condition:
		analysis_view = {
			"running": analysis_running,
			"status": analysis_status,
			"error": analysis_error,
			"total": analysis_total,
			"job_id": analysis_job_id,
			"video_file": analysis_video_file,
			"model_path": analysis_model_path,
		}

	return render_template_string("""
	<h1>PC Agent Analysis</h1>
	<p>Run this service in background and control analysis from this page.</p>

	{% if message %}
		<p><strong>{{ message }}</strong></p>
	{% endif %}

	<form action="/analyze" method="post" enctype="multipart/form-data" style="margin-bottom: 1rem;">
		<label>Video (.mp4):</label><br>
		<input type="file" name="video_file" accept=".mp4,video/mp4" required><br><br>

		<label>Model (.pt):</label><br>
		<input type="file" name="model_file" accept=".pt" required><br><br>

		<button type="submit">Start Analysis</button>
	</form>

	<p>Status: <strong>{{ analysis.status|upper }}</strong></p>
	{% if analysis.video_file %}
		<p>Current video: {{ analysis.video_file }}</p>
	{% endif %}
	{% if analysis.model_path %}
		<p>Current model: {{ analysis.model_path }}</p>
	{% endif %}
	{% if analysis.error %}
		<p style="color: red;">Error: {{ analysis.error }}</p>
	{% endif %}

	{% if analysis.status in ['running', 'completed'] and analysis.job_id %}
		<img src="/analysis_feed?job={{ analysis.job_id }}" style="height: 1080px; width: auto; border: 1px solid #ddd;">
		<p>Total crossed: {{ analysis.total }}</p>
		{% if analysis.status == 'running' %}
			<p>Analysis is running. Keep this page open to watch live output.</p>
		{% endif %}
	{% endif %}
	""", message=message, analysis=analysis_view)


@app.route("/analyze", methods=["POST"])
def analyze_from_browser():
	os.makedirs(UPLOAD_DIR, exist_ok=True)
	os.makedirs(MODEL_UPLOAD_DIR, exist_ok=True)
	file_obj = request.files.get("video_file")
	if file_obj is None or file_obj.filename is None or file_obj.filename.strip() == "":
		return redirect(url_for("index", message="No file selected"))

	safe_name = secure_filename(file_obj.filename)
	if not _is_allowed_video(safe_name):
		return redirect(url_for("index", message="Only .mp4 files are supported"))

	stored_name = f"{int(time.time())}_{safe_name}"
	stored_path = os.path.join(UPLOAD_DIR, stored_name)
	file_obj.save(stored_path)

	model_file_obj = request.files.get("model_file")
	if model_file_obj is None or model_file_obj.filename is None or model_file_obj.filename.strip() == "":
		try:
			os.remove(stored_path)
		except OSError:
			pass
		return redirect(url_for("index", message="No model selected"))

	model_safe_name = secure_filename(model_file_obj.filename)
	if not _is_allowed_model(model_safe_name):
		try:
			os.remove(stored_path)
		except OSError:
			pass
		return redirect(url_for("index", message="Only .pt model files are supported"))

	stored_model_name = f"{int(time.time() * 1000)}_{model_safe_name}"
	stored_model_path = os.path.join(MODEL_UPLOAD_DIR, stored_model_name)
	model_file_obj.save(stored_model_path)

	if not os.path.isfile(stored_model_path):
		try:
			os.remove(stored_path)
		except OSError:
			pass
		return redirect(url_for("index", message="Failed to store model file"))

	ok, status_message = start_analysis_job(stored_path, safe_name, stored_model_path)
	if not ok:
		try:
			os.remove(stored_path)
		except OSError:
			pass
		try:
			os.remove(stored_model_path)
		except OSError:
			pass
	return redirect(url_for("index", message=status_message))


@app.route("/analysis_feed")
def analysis_feed():
	job_id = request.args.get("job", "")
	with analysis_condition:
		if not analysis_job_id:
			return "No analysis job", 404
		if job_id and job_id != analysis_job_id:
			return "Analysis job has changed", 410
		active_job = analysis_job_id
	return Response(generate_analysis_frames(active_job), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/analysis_status")
def analysis_status_route():
	with analysis_condition:
		return jsonify({
			"running": analysis_running,
			"status": analysis_status,
			"error": analysis_error,
			"total": analysis_total,
			"job_id": analysis_job_id,
			"file": analysis_video_file,
			"model": analysis_model_path,
		})


@app.route("/cancel_analysis", methods=["POST"])
def cancel_analysis_route():
	ok, message = cancel_analysis_job()
	status_code = 200 if ok else 409
	return jsonify({"ok": ok, "message": message}), status_code


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run YOLO + Norfair analysis")
	parser.add_argument("--source", default=DEFAULT_SOURCE, help="Path to input video")
	parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to YOLO model")
	parser.add_argument("--mode", choices=["gui", "api"], default="api", help="Run OpenCV window or Flask API")
	parser.add_argument("--host", default="0.0.0.0", help="Host to bind in API mode")
	parser.add_argument("--port", type=int, default=5050, help="Port to bind in API mode")
	args = parser.parse_args()

	if args.mode == "api":
		app.run(host=args.host, port=args.port, threaded=True)
	else:
		run_analysis(source=args.source, model_path=_resolve_model_path(args.model))