import os
import subprocess
import threading
import time
import importlib
import io
import zipfile
from flask import Flask, render_template_string, send_from_directory, request, redirect, url_for, Response
from flask import jsonify
from flask import send_file
from werkzeug.utils import secure_filename

#http://<your-pi-ip>:5000

# Camera imports
try:
	Picamera2 = importlib.import_module("picamera2").Picamera2
	MJPEGEncoder = importlib.import_module("picamera2.encoders").MJPEGEncoder
	FileOutput = importlib.import_module("picamera2.outputs").FileOutput
	CAMERA_AVAILABLE = True
except Exception:
	Picamera2 = None
	MJPEGEncoder = None
	FileOutput = None
	CAMERA_AVAILABLE = False

app = Flask(__name__)

SCRIPT_PATH = "/home/plasticpi/camSaveCustom.py"
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_PATH), "data")
PROGRAM_DIR = os.path.join(os.path.dirname(SCRIPT_PATH), "download_program")
PROGRAM_MAC_DIR = os.path.join(os.path.dirname(SCRIPT_PATH), "download_program_mac")
PROGRAM_WINDOWS_DIR = os.path.join(os.path.dirname(SCRIPT_PATH), "download_program_windows")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
DEFAULT_FILENAME = "video.mp4"
DEFAULT_FPS = 60
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 1080
DEFAULT_RESOLUTION_KEY = "640x1080"
DEFAULT_ANALYSIS_MODEL = "granulatModels/granulat177def.pt"
ALLOWED_ANALYSIS_EXTENSIONS = {".mp4"}

ALLOWED_RESOLUTIONS = {
	"640x1080": {"size": (640, 1080)},
	"1280x1080": {"size": (1280, 1080)},
	"1920x1080": {"size": (1920, 1080)},
	"1536x864 120fps": {"size": (1536, 864), "fps": 120},
}

# ---------------- CAMERA STREAM ----------------
class StreamingOutput(io.BufferedIOBase):
	def __init__(self):
		self.frame = None
		self.condition = threading.Condition()

	def write(self, buf):
		with self.condition:
			self.frame = buf
			self.condition.notify_all()

output = StreamingOutput()
picam2 = None
preview_running = False
preview_lock = threading.Lock()

# ---------------- ANALYSIS STATE ----------------
analysis_condition = threading.Condition()
analysis_running = False
analysis_status = "idle"
analysis_error = ""
analysis_frame = None
analysis_frame_seq = 0
analysis_total = 0
analysis_job_id = ""
analysis_file = ""
analysis_model = ""


def start_preview_stream():
	global picam2, preview_running
	if not CAMERA_AVAILABLE:
		raise RuntimeError("picamera2 is not available on this machine")
	with preview_lock:
		if preview_running:
			return
		last_error = None
		for _ in range(15):
			try:
				picam2 = Picamera2()
				picam2.configure(picam2.create_video_configuration(
					main={"size": (1920, 1080)},
					controls={"FrameRate": 10}
				))
				picam2.start_recording(MJPEGEncoder(), FileOutput(output))
				preview_running = True
				return
			except Exception as err:
				last_error = err
				# Ensure partially initialized camera objects are released before retry.
				try:
					if picam2 is not None:
						picam2.close()
				except Exception:
					pass
				picam2 = None
				preview_running = False
				time.sleep(0.2)
		raise RuntimeError(f"Preview camera start failed after retries: {last_error}")


def stop_preview_stream():
	global picam2, preview_running
	with preview_lock:
		if not preview_running or picam2 is None:
			return
		picam2.stop_recording()
		picam2.close()
		picam2 = None
		preview_running = False

if CAMERA_AVAILABLE:
	try:
		start_preview_stream()
	except Exception as err:
		print(f"Preview startup skipped: {err}")


def generate_frames():
	while True:
		with output.condition:
			output.condition.wait()
			frame = output.frame
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def _is_allowed_analysis_file(filename):
	_, ext = os.path.splitext(filename or "")
	return ext.lower() in ALLOWED_ANALYSIS_EXTENSIONS


def _list_analysis_models():
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

	if DEFAULT_ANALYSIS_MODEL not in model_paths:
		model_paths.append(DEFAULT_ANALYSIS_MODEL)

	return sorted(set(model_paths))


def _resolve_model_path(model_path):
	candidate = (model_path or "").strip()
	if not candidate:
		candidate = DEFAULT_ANALYSIS_MODEL

	if os.path.isabs(candidate):
		return candidate

	return os.path.join(os.path.dirname(__file__), candidate)


def _analysis_worker(input_path, model_path, job_id):
	global analysis_running, analysis_status, analysis_error
	global analysis_frame, analysis_frame_seq, analysis_total

	try:
		from pc_agent import analyze_video_stream

		for update in analyze_video_stream(input_path, model_path=model_path):
			with analysis_condition:
				if analysis_job_id != job_id:
					return
				analysis_frame = update.get("jpeg")
				analysis_total = int(update.get("total", analysis_total))
				analysis_frame_seq += 1
				analysis_condition.notify_all()

		with analysis_condition:
			if analysis_job_id == job_id:
				analysis_running = False
				analysis_status = "completed"
				analysis_condition.notify_all()
	except Exception as err:
		with analysis_condition:
			if analysis_job_id == job_id:
				analysis_running = False
				analysis_status = "error"
				analysis_error = str(err)
				analysis_condition.notify_all()


def start_analysis_job(input_path, original_filename, model_path):
	global analysis_running, analysis_status, analysis_error
	global analysis_frame, analysis_frame_seq, analysis_total
	global analysis_job_id, analysis_file, analysis_model

	with analysis_condition:
		if analysis_running:
			return False, "Analysis already running"

		analysis_job_id = str(int(time.time() * 1000))
		analysis_file = original_filename
		analysis_model = model_path
		analysis_running = True
		analysis_status = "running"
		analysis_error = ""
		analysis_frame = None
		analysis_frame_seq = 0
		analysis_total = 0

	threading.Thread(target=_analysis_worker, args=(input_path, model_path, analysis_job_id), daemon=True).start()
	return True, "Analysis started"


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

# ---------------- PROCESS MANAGEMENT ----------------
def is_script_running():
	result = subprocess.run(["pgrep", "-f", SCRIPT_PATH], stdout=subprocess.PIPE)
	return result.stdout.strip() != b""


def normalize_filename(filename):
	name = os.path.basename((filename or "").strip())
	if not name:
		name = DEFAULT_FILENAME
	if not name.lower().endswith(".mp4"):
		name = f"{name}.mp4"
	return name


def parse_fps(value):
	try:
		fps = int(value)
		return fps if fps > 0 else DEFAULT_FPS
	except (TypeError, ValueError):
		return DEFAULT_FPS


def parse_resolution(value):
	preset = ALLOWED_RESOLUTIONS.get(value)
	if preset is None:
		default_preset = ALLOWED_RESOLUTIONS.get(DEFAULT_RESOLUTION_KEY, {})
		size = default_preset.get("size", (DEFAULT_WIDTH, DEFAULT_HEIGHT))
		return size[0], size[1], None

	size = preset.get("size", (DEFAULT_WIDTH, DEFAULT_HEIGHT))
	fps_override = preset.get("fps")
	return size[0], size[1], fps_override


def start_script(filename=DEFAULT_FILENAME, fps=DEFAULT_FPS, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
	if not is_script_running():
		stop_preview_stream()
		subprocess.Popen([
			"python3",
			SCRIPT_PATH,
			filename,
			str(fps),
			str(width),
			str(height),
		])


def stop_script():
	subprocess.run(["pkill", "-f", SCRIPT_PATH])


def wait_for_script_stop(timeout_seconds=6):
	deadline = time.monotonic() + timeout_seconds
	while time.monotonic() < deadline:
		if not is_script_running():
			return True
		time.sleep(0.2)
	return not is_script_running()


def get_processes():
	result = subprocess.run(["ps", "aux"], stdout=subprocess.PIPE)
	return result.stdout.decode()

# ---------------- ROUTES ----------------
@app.route('/')
def index():
	os.makedirs(DATA_DIR, exist_ok=True)
	files = sorted(os.listdir(DATA_DIR))
	running = is_script_running()
	processes = get_processes()
	message = request.args.get("message", "")
	selected_model = request.args.get("model", DEFAULT_ANALYSIS_MODEL)
	selected_resolution = request.args.get("resolution", DEFAULT_RESOLUTION_KEY)
	available_models = _list_analysis_models()
	resolution_options = list(ALLOWED_RESOLUTIONS.keys())
	if selected_model not in available_models:
		selected_model = DEFAULT_ANALYSIS_MODEL
	if selected_resolution not in ALLOWED_RESOLUTIONS:
		selected_resolution = DEFAULT_RESOLUTION_KEY
	camera_active = CAMERA_AVAILABLE and not running
	with analysis_condition:
		analysis_view = {
			"running": analysis_running,
			"status": analysis_status,
			"error": analysis_error,
			"total": analysis_total,
			"job_id": analysis_job_id,
			"file": analysis_file,
			"model": analysis_model,
		}

	return render_template_string("""
	<h1>Raspberry Pi Control Panel</h1>

	{% if message %}
		<p><strong>{{ message }}</strong></p>
	{% endif %}

	<h2>camSave.py Status: {{ 'RUNNING' if running else 'STOPPED' }}</h2>
	<form action="/start" method="post" style="margin-bottom: 1rem;">
		<label>Filename (.mp4):</label><br>
		<input name="filename" type="text" value="video.mp4" required><br><br>

		<label>FPS:</label><br>
		<input name="fps" type="number" min="1" value="60" required><br><br>

		<label>Resolution:</label><br>
		<select name="resolution">
			{% for resolution in resolution_options %}
				<option value="{{ resolution }}" {% if resolution == selected_resolution %}selected{% endif %}>{{ resolution }}</option>
			{% endfor %}
		</select><br><br>

		<button>Start Script</button>
	</form>
	<form action="/stop" method="post"><button>Stop Script</button></form>

	<h2>Camera Feed (only when stopped)</h2>
	{% if camera_active %}
		<img src="/video_feed" width="640">
	{% elif not camera_available %}
		<p>Camera preview unavailable: picamera2 is not installed on this machine.</p>
	{% else %}
		<p>Camera disabled while script is running.</p>
	{% endif %}

	<hr>
	<h2>Analyze Local MP4 with YOLO</h2>
	<p>Analysis is executed on your local machine by pc_agent. This page only controls and displays the live output stream.</p>
	<form id="analyze-form" action="/analyze" method="post" enctype="multipart/form-data" style="margin-bottom: 1rem;">
		<label>Local pc_agent URL:</label><br>
		<input id="agent-url" type="text" value="http://127.0.0.1:5050" style="min-width: 320px;"><br><br>
		<label>Select MP4 file:</label><br>
		<input type="file" name="video_file" accept=".mp4,video/mp4" required><br><br>
		<label>Select model (.pt):</label><br>
		<input type="file" name="model_file" accept=".pt" required><br><br>
		<button id="analyze-btn" type="submit">Analyze</button>
		<button id="cancel-analyze-btn" type="button">Stop Analysis</button>
	</form>

	<p id="analysis-hint" style="display:none;"><strong>Uploading and starting analysis...</strong></p>
	<p>Local pc_agent: <strong id="agent-status">CHECKING...</strong></p>
	<p>Status: <strong id="analysis-status">{{ analysis.status|upper }}</strong></p>
	{% if analysis.file %}
		<p id="analysis-file">Current file: {{ analysis.file }}</p>
	{% else %}
		<p id="analysis-file" style="display:none;"></p>
	{% endif %}
	{% if analysis.model %}
		<p id="analysis-model">Current model: {{ analysis.model }}</p>
	{% else %}
		<p id="analysis-model" style="display:none;"></p>
	{% endif %}
	{% if analysis.error %}
		<p id="analysis-error" style="color: red;">Error: {{ analysis.error }}</p>
	{% else %}
		<p id="analysis-error" style="color: red; display:none;"></p>
	{% endif %}
	{% if analysis.status in ['running', 'completed'] and analysis.job_id %}
		<img id="analysis-stream" src="/analysis_feed?job={{ analysis.job_id }}" style="height: 1080px; width: auto; border: 1px solid #ddd;">
		<p id="analysis-total">Total crossed: {{ analysis.total }}</p>
		{% if analysis.status == 'running' %}
			<p id="analysis-running-note">Analysis is running. Keep this page open to watch the stream.</p>
		{% else %}
			<p id="analysis-running-note" style="display:none;"></p>
		{% endif %}
	{% else %}
		<img id="analysis-stream" src="" style="display:none; height: 720px; width: auto; border: 1px solid #ddd;">
		<p id="analysis-total">Total crossed: {{ analysis.total }}</p>
		<p id="analysis-running-note" style="display:none;"></p>
	{% endif %}

	<script>
	const formEl = document.getElementById('analyze-form');
	const agentUrlEl = document.getElementById('agent-url');
	const btnEl = document.getElementById('analyze-btn');
	const cancelBtnEl = document.getElementById('cancel-analyze-btn');
	const hintEl = document.getElementById('analysis-hint');
	const agentStatusEl = document.getElementById('agent-status');
	const statusEl = document.getElementById('analysis-status');
	const fileEl = document.getElementById('analysis-file');
	const modelEl = document.getElementById('analysis-model');
	const errEl = document.getElementById('analysis-error');
	const totalEl = document.getElementById('analysis-total');
	const streamEl = document.getElementById('analysis-stream');
	const runningNoteEl = document.getElementById('analysis-running-note');
	let currentJobId = {{ analysis.job_id|tojson }};
	let currentAgentBase = 'http://127.0.0.1:5050';

	if (agentUrlEl) {
		const savedAgentUrl = localStorage.getItem('pcAgentBaseUrl');
		if (savedAgentUrl) {
			agentUrlEl.value = savedAgentUrl;
		}
	}

	function getAgentBase() {
		const fallback = 'http://127.0.0.1:5050';
		const raw = (agentUrlEl && agentUrlEl.value ? agentUrlEl.value : fallback).trim();
		const normalized = raw.replace(/\/+$/, '');
		return normalized || fallback;
	}

	function setAgentStatus(text, color) {
		if (!agentStatusEl) return;
		agentStatusEl.textContent = text;
		agentStatusEl.style.color = color || '';
	}

	async function fetchWithTimeout(url, options = {}, timeoutMs = 2500) {
		const controller = new AbortController();
		const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
		try {
			return await fetch(url, {
				...options,
				signal: controller.signal,
			});
		} finally {
			clearTimeout(timeoutId);
		}
	}

	async function probeAgent() {
		currentAgentBase = getAgentBase();
		try {
			const resp = await fetchWithTimeout(`${currentAgentBase}/analysis_status`, { cache: 'no-store', mode: 'cors' }, 2500);
			if (!resp.ok) {
				setAgentStatus(`OFFLINE (HTTP ${resp.status})`, 'red');
				return { online: false, response: null };
			}
			setAgentStatus('ONLINE', 'green');
			return { online: true, response: resp };
		} catch (err) {
			setAgentStatus('OFFLINE', 'red');
			return { online: false, response: null };
		}
	}

	if (formEl) {
		formEl.addEventListener('submit', async (event) => {
			event.preventDefault();
			currentAgentBase = getAgentBase();
			localStorage.setItem('pcAgentBaseUrl', currentAgentBase);

			if (hintEl) hintEl.style.display = 'block';
			if (hintEl) hintEl.textContent = 'Uploading to local pc_agent and starting analysis...';
			if (btnEl) btnEl.disabled = true;

			const formData = new FormData(formEl);
			try {
				const probe = await probeAgent();
				if (!probe.online) {
					throw new Error('pc_agent is not reachable');
				}

				const resp = await fetch(`${currentAgentBase}/analyze`, {
					method: 'POST',
					body: formData,
					mode: 'cors',
				});
				if (!resp.ok) {
					throw new Error(`local pc_agent returned HTTP ${resp.status}`);
				}
				await pollAnalysisState();
			} catch (err) {
				if (statusEl) statusEl.textContent = 'ERROR';
				if (errEl) {
					errEl.textContent = `Error: Could not reach local pc_agent at ${currentAgentBase}. ${err.message}`;
					errEl.style.display = 'block';
				}
				if (hintEl) hintEl.style.display = 'none';
				if (btnEl) btnEl.disabled = false;
			}
		});
	}

	if (cancelBtnEl) {
		cancelBtnEl.addEventListener('click', async () => {
			currentAgentBase = getAgentBase();
			localStorage.setItem('pcAgentBaseUrl', currentAgentBase);
			cancelBtnEl.disabled = true;
			try {
				const probe = await probeAgent();
				if (!probe.online) {
					throw new Error('pc_agent is not reachable');
				}

				const resp = await fetch(`${currentAgentBase}/cancel_analysis`, {
					method: 'POST',
					mode: 'cors',
				});
				if (!resp.ok && resp.status !== 409) {
					throw new Error(`local pc_agent returned HTTP ${resp.status}`);
				}
				await pollAnalysisState();
			} catch (err) {
				if (errEl) {
					errEl.textContent = `Error: Could not send cancel to local pc_agent at ${currentAgentBase}. ${err.message}`;
					errEl.style.display = 'block';
				}
			} finally {
				cancelBtnEl.disabled = false;
			}
		});
	}

	function updateStatusView(state) {
		if (statusEl) statusEl.textContent = (state.status || 'idle').toUpperCase();
		if (fileEl) {
			if (state.file) {
				fileEl.textContent = `Current file: ${state.file}`;
				fileEl.style.display = 'block';
			} else {
				fileEl.style.display = 'none';
			}
		}
		if (modelEl) {
			if (state.model) {
				modelEl.textContent = `Current model: ${state.model}`;
				modelEl.style.display = 'block';
			} else {
				modelEl.style.display = 'none';
			}
		}
		if (totalEl) totalEl.textContent = `Total crossed: ${state.total || 0}`;
		if (errEl) {
			if (state.error) {
				errEl.textContent = `Error: ${state.error}`;
				errEl.style.display = 'block';
			} else {
				errEl.style.display = 'none';
			}
		}

		if (runningNoteEl) {
			if (state.status === 'running') {
				runningNoteEl.textContent = 'Analysis is running locally. Keep this page open to watch the stream.';
				runningNoteEl.style.display = 'block';
			} else {
				runningNoteEl.style.display = 'none';
			}
		}

		if (streamEl && state.job_id && (state.status === 'running' || state.status === 'completed')) {
			if (currentJobId !== state.job_id) {
				streamEl.src = `${currentAgentBase}/analysis_feed?job=${encodeURIComponent(state.job_id)}&t=${Date.now()}`;
				currentJobId = state.job_id;
			}
			streamEl.style.display = 'block';
		} else if (streamEl && state.status === 'idle') {
			streamEl.style.display = 'none';
		}

		if (btnEl && state.status !== 'running') {
			btnEl.disabled = false;
		}
		if (cancelBtnEl) {
			cancelBtnEl.disabled = state.status !== 'running';
		}
		if (hintEl && state.status !== 'idle') {
			hintEl.style.display = 'none';
		}
	}

	async function pollAnalysisState() {
		try {
			const probe = await probeAgent();
			if (!probe.online || !probe.response) {
				if (statusEl) statusEl.textContent = 'LOCAL AGENT OFFLINE';
				if (errEl) {
					errEl.textContent = `Error: Could not reach local pc_agent at ${currentAgentBase}`;
					errEl.style.display = 'block';
				}
				if (btnEl) btnEl.disabled = false;
				if (cancelBtnEl) cancelBtnEl.disabled = true;
				return;
			}

			const state = await probe.response.json();
			updateStatusView(state);
		} catch (err) {
			setAgentStatus('OFFLINE', 'red');
			if (statusEl) statusEl.textContent = 'LOCAL AGENT OFFLINE';
			if (errEl) {
				errEl.textContent = `Error: Could not reach local pc_agent at ${currentAgentBase}`;
				errEl.style.display = 'block';
			}
		}
	}

	setInterval(pollAnalysisState, 1000);
	pollAnalysisState();
	</script>

	<h2>Files</h2>
	<ul>
	{% for f in files %}
		<li>
			<a href="/download/{{f}}">{{f}}</a>
			<form action="/delete/{{f}}" method="post" style="display:inline; margin-left: 0.5rem;">
				<button type="submit" onclick="return confirm('Delete {{f}}?')">Delete</button>
			</form>
		</li>
	{% endfor %}
	</ul>

	<h2>Processes</h2>
	<pre style="height:300px; overflow:auto;">{{processes}}</pre>

	<h2>Program Folder</h2>
	<div style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">
		<form action="/download_program" method="get" style="margin:0;">
			<button type="submit">Download Sourcecode</button>
		</form>
		<form action="/download_program_mac" method="get" style="margin:0;">
			<button type="submit">Download Mac</button>
		</form>
		<form action="/download_program_windows" method="get" style="margin:0;">
			<button type="submit">Download Windows</button>
		</form>
	</div>
	""", files=files, running=running, processes=processes, analysis=analysis_view, message=message, camera_available=CAMERA_AVAILABLE, camera_active=camera_active, available_models=available_models, selected_model=selected_model, resolution_options=resolution_options, selected_resolution=selected_resolution)

@app.route('/download/<path:filename>')
def download_file(filename):
	return send_from_directory(DATA_DIR, filename, as_attachment=True)


def _download_program_folder(folder_path, zip_name, missing_message):
	if not os.path.isdir(folder_path):
		return redirect(url_for('index', message=missing_message))

	zip_buffer = io.BytesIO()
	with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zipf:
		for root, _, files in os.walk(folder_path):
			for filename in files:
				full_path = os.path.join(root, filename)
				arcname = os.path.relpath(full_path, folder_path)
				zipf.write(full_path, arcname)

	zip_buffer.seek(0)
	return send_file(
		zip_buffer,
		mimetype='application/zip',
		as_attachment=True,
		download_name=zip_name,
	)


@app.route('/download_program')
def download_program():
	return _download_program_folder(PROGRAM_DIR, 'sourcecode.zip', 'Sourcecode folder not found on Raspberry Pi')


@app.route('/download_program_mac')
def download_program_mac():
	return _download_program_folder(PROGRAM_MAC_DIR, 'program_mac.zip', 'Mac program folder not found on Raspberry Pi')


@app.route('/download_program_windows')
def download_program_windows():
	return _download_program_folder(PROGRAM_WINDOWS_DIR, 'program_windows.zip', 'Windows program folder not found on Raspberry Pi')

@app.route('/start', methods=['POST'])
def start():
	filename = normalize_filename(request.form.get("filename", DEFAULT_FILENAME))
	fps = parse_fps(request.form.get("fps", DEFAULT_FPS))
	resolution_key = request.form.get("resolution", DEFAULT_RESOLUTION_KEY)
	width, height, fps_override = parse_resolution(resolution_key)
	if fps_override is not None:
		fps = fps_override
	start_script(filename=filename, fps=fps, width=width, height=height)
	return redirect(url_for('index', resolution=resolution_key))

@app.route('/stop', methods=['POST'])
def stop():
	stop_script()
	if wait_for_script_stop():
		try:
			start_preview_stream()
		except Exception as err:
			print(f"Preview restart failed: {err}")
	return redirect(url_for('index'))


@app.route('/analyze', methods=['POST'])
def analyze():
	os.makedirs(UPLOAD_DIR, exist_ok=True)
	file_obj = request.files.get("video_file")
	if file_obj is None or file_obj.filename is None or file_obj.filename.strip() == "":
		return redirect(url_for('index', message='No file selected'))

	safe_name = secure_filename(file_obj.filename)
	if not _is_allowed_analysis_file(safe_name):
		return redirect(url_for('index', message='Only .mp4 files are supported'))

	stored_name = f"{int(time.time())}_{safe_name}"
	stored_path = os.path.join(UPLOAD_DIR, stored_name)
	file_obj.save(stored_path)

	model_choice = request.form.get("model_path", DEFAULT_ANALYSIS_MODEL)
	resolved_model = _resolve_model_path(model_choice)
	if not os.path.isfile(resolved_model):
		try:
			os.remove(stored_path)
		except OSError:
			pass
		return redirect(url_for('index', message='Model not found', model=model_choice))

	ok, status_message = start_analysis_job(stored_path, safe_name, resolved_model)
	if not ok:
		try:
			os.remove(stored_path)
		except OSError:
			pass
	return redirect(url_for('index', message=status_message, model=model_choice))


@app.route('/delete/<path:filename>', methods=['POST'])
def delete_file(filename):
	safe_name = os.path.basename(filename)
	file_path = os.path.join(DATA_DIR, safe_name)
	if os.path.isfile(file_path):
		os.remove(file_path)
	return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
	if not CAMERA_AVAILABLE:
		return "Camera preview unavailable on this machine", 503
	if is_script_running():
		return "Camera in use", 403
	if not preview_running:
		try:
			start_preview_stream()
		except Exception as err:
			return f"Camera preview unavailable: {err}", 503
	return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/analysis_feed')
def analysis_feed():
	job_id = request.args.get("job", "")
	with analysis_condition:
		if not analysis_job_id:
			return "No analysis job", 404
		if job_id and job_id != analysis_job_id:
			return "Analysis job has changed", 410
		active_job = analysis_job_id
	return Response(generate_analysis_frames(active_job), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/analysis_status')
def analysis_status_route():
	with analysis_condition:
		return jsonify({
			"running": analysis_running,
			"status": analysis_status,
			"error": analysis_error,
			"total": analysis_total,
			"job_id": analysis_job_id,
			"file": analysis_file,
			"model": analysis_model,
		})

# ---------------- MAIN ----------------
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, threaded=True)
