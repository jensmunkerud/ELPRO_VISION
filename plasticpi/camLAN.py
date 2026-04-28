from flask import Flask, Response
import io
import threading
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

picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"size": (1280, 720)},
        controls={"FrameRate": 60}
    )
)

output = StreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(output))

picam2.set_controls({
        "AfMode": 0,
        "LensPosition": 10,
        "ExposureTime": 10000,   # microseconds (20 ms)
        "AnalogueGain": 1.0,     # ISO multiplier
        "AwbMode": 0,            # optional: turn off auto white balance
        "AeEnable": False         # turn off auto-exposure
})

@app.route('/')
def index():
    return '''
    <html>
        <head><title>Pi Camera Stream</title></head>
        <body>
            <h1>Pi Camera Stream</h1>
            <img src="/video_feed" width="960">
        </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(output),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
