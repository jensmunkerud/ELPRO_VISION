from ultralytics import YOLO

model = YOLO("granulatModels/granulat177def.pt")

results = model(
	source="test_videos/seint.mp4",
	show=True
)
