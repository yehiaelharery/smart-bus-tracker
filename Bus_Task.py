import cv2
import torch
import numpy as np
from ultralytics import YOLO
from boxmot.tracker_zoo import create_tracker
from pathlib import Path

# Parameters
TICKET_PRICE = 15
VIDEO_SOURCE = "C:/Users/Lenovo/Downloads/video6.mp4"
LINE_POSITION = 300
LINE_OFFSET = 10
MIN_CONFIDENCE = 0.3

# Paths to config and ReID weights
REID_WEIGHTS_PATH = Path(
    "C:/Users/Lenovo/log2/my_reid_model/Bus_reid_model.pt"
)
TRACKING_CONFIG_PATH = Path(
    "C:/Users/Lenovo/boxmot/boxmot/configs/trackers/botsort.yaml"
)

# Load YOLOv8
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8s.pt").to(device)
model.fuse()

# Load StrongSORT / BoTSORT tracker
tracker = create_tracker(
    "botsort",
    TRACKING_CONFIG_PATH,
    reid_weights=REID_WEIGHTS_PATH,
    reid_model='osnet_x1_0',
    device=device
)

# Initialize state
already_counted_ids = set()
passenger_count = 0

# Start video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Get video properties for VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
# 'XVID' is a good codec for .avi files
# 'mp4v' or 'H264' are common for .mp4, but may require specific installations
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('bus_counter_output.avi', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detection (only people = class 0)
    results = model(frame, classes=[0], verbose=False)[0]
    
    detections = []
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls = box
        if score > MIN_CONFIDENCE:
            detections.append([x1, y1, x2, y2, score, 0])  # person class_id = 0

    detections_np = np.array(detections) if detections else np.empty((0, 6))

    # Tracking with StrongSORT
    tracks = tracker.update(detections_np, frame)

    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        
        # Calculate center of person
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        # Crossing the line logic
        if LINE_POSITION - LINE_OFFSET < cy < LINE_POSITION + LINE_OFFSET:
            if track_id not in already_counted_ids:
                already_counted_ids.add(track_id)
                passenger_count += 1
                print(f"Passenger Counted! ID={track_id} Total={passenger_count}")
        
        # Draw person bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw virtual line
    cv2.line(frame, (0, LINE_POSITION), (frame.shape[1], LINE_POSITION), (0, 0, 255), 2)

    # Show passenger count & fare
    fare = passenger_count * TICKET_PRICE
    cv2.putText(frame, f"Count: {passenger_count} Fare: {fare} EGP",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write the frame to the output video file
    out.write(frame)

    # Display result
    cv2.imshow("Passenger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything when the job is done
cap.release()
out.release()
cv2.destroyAllWindows()