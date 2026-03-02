from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Run live inference from default webcam
results = model.track(source=0, show=True, stream=True)

for r in results:
    pass