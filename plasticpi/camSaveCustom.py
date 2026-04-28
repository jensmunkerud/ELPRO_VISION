from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import sys
import os
import time
import signal

picam2 = Picamera2()

def parse_int_arg(index, default_value):
	try:
		value = int(sys.argv[index])
		return value if value > 0 else default_value
	except (IndexError, ValueError):
		return default_value


raw_name = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
name = os.path.basename(raw_name.strip()) or "video.mp4"
if not name.lower().endswith(".mp4"):
	name = f"{name}.mp4"

fps = parse_int_arg(2, 120)
width = parse_int_arg(3, 640)
height = parse_int_arg(4, 480)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
filename = os.path.join(DATA_DIR, name)

os.makedirs(DATA_DIR, exist_ok=True)

video_config = picam2.create_video_configuration(
		main={"size": (width, height)},
		controls={"FrameRate": fps}
)

picam2.configure(video_config)

encoder = H264Encoder(bitrate=15000000)
output = FfmpegOutput(filename)

picam2.start_recording(encoder, output)

running = True


def request_stop(signum, frame):
	global running
	running = False


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)

picam2.set_controls({
		"AfMode": 0,
		"LensPosition": 10,
		"ExposureTime": 5000,
		"AnalogueGain": 200.0,
		"AwbMode": 1,
		"AeEnable": False
})

print(f"Recording to {filename}")
print(f"Settings: {width}x{height} @ {fps} FPS")
print("Press Ctrl+C to stop")

try:
		while running:
				time.sleep(1)

except KeyboardInterrupt:
		print("\nStopping recording...")

finally:
		picam2.stop_recording()
		print(f"Saved video to {filename}")
