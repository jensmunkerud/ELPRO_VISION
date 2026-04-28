from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import sys
import os
import time

picam2 = Picamera2()

name = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
filename = os.path.join("data", name)

os.makedirs("data", exist_ok=True)

video_config = picam2.create_video_configuration(
	main={"size": (1920, 1080)},
	controls={"FrameRate": 60}
)

picam2.configure(video_config)

encoder = H264Encoder(bitrate=15000000)
output = FfmpegOutput(filename)

picam2.start_recording(encoder, output)

picam2.set_controls({
	"AfMode": 0,
	"LensPosition": 10,
	"ExposureTime": 5000,
	"AnalogueGain": 200.0,
	"AwbMode": 1,
	"AeEnable": False
})

print(f"Recording to {filename}")
print("Press Ctrl+C to stop")

try:
	while True:
		time.sleep(1)

except KeyboardInterrupt:
	print("\nStopping recording...")

finally:
	picam2.stop_recording()
	print(f"Saved video to {filename}")
