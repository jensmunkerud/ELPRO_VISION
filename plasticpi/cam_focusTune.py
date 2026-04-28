import io
import sys
import termios
import tty
import threading
from flask import Flask, Response
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

app = Flask(__name__)

class StreamingOutput(io.BufferedIOBase):
	def __init__(self):
		self.frame = None
		self.condition = threading.Condition()

	def write(self, buf):
		with self.condition:
			self.frame = buf
			self.condition.notify_all()


def generate_frames(output):
	while True:
		with output.condition:
			output.condition.wait()
			frame = output.frame

		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def keyboard_focus_control(picam2):
	fd = sys.stdin.fileno()
	old_settings = termios.tcgetattr(fd)

	focus = 2.0

	try:
		tty.setraw(fd)

		while True:
			key = sys.stdin.read(1)

			if key == "w":
				focus = min(focus + 1, 20)
				picam2.set_controls({"LensPosition": focus})
				print(f"\nFocus: {focus}")

			elif key == "s":
				focus = max(focus - 1, 0)
				picam2.set_controls({"LensPosition": focus})
				print(f"\nFocus: {focus}")

			elif key == "q":
				print("\nExiting...")
				break

	finally:
		termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# Camera setup
picam2 = Picamera2()

picam2.configure(
	picam2.create_video_configuration(
		main={"size": (1920, 1080)},
		controls={"FrameRate": 10}
	)
)

output = StreamingOutput()

picam2.start()

# enable manual focus
picam2.set_controls({
	"AfMode": 0,
	"LensPosition": 2.0
})

picam2.start_recording(MJPEGEncoder(), FileOutput(output))


@app.route('/video_feed')
def video_feed():
	return Response(
		generate_frames(output),
		mimetype='multipart/x-mixed-replace; boundary=frame'
	)


def run_flask():
	app.run(host='0.0.0.0', port=5000, threaded=True)


if __name__ == '__main__':
	threading.Thread(target=run_flask, daemon=True).start()

	print("Controls:")
	print("w = increase focus")
	print("s = decrease focus")
	print("q = quit\n")

	keyboard_focus_control(picam2)
