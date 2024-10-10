# pip install ultralytics
from ultralytics import YOLO

model = YOLO("./runs/detect/pitch_detection_v5/weights/best.pt")

# Set the baseball detection confidence with 'conf'
results = model.predict("./pitcher_vids/degrom1.mp4", show=True, save=True, conf=0.15)
