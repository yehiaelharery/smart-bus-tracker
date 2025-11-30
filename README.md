# Bus Passenger Counter

An AI-powered system that detects, tracks, and counts passengers on a bus in real-time using YOLOv8 for person detection and BoTSORT/StrongSORT for tracking. The system also calculates the total fare collected based on a fixed ticket price.

---

## Features

- Real-time passenger detection using YOLOv8
- Multi-object tracking with StrongSORT / BoTSORT
- Automatic passenger counting when crossing a virtual line
- Fare calculation based on passenger count
- Annotated output video with bounding boxes, IDs, counts, and fare

---

## Demo

A demo video is included in this repository: [`Bus_Reid_Project.mp4`](Bus_Reid_Project4.mp4). You can watch it to see the system in action.

---

## Installation & Setup

1. Clone the repository:

git clone https://github.com/your-username/bus-passenger-counter.git  
cd bus-passenger-counter

2. (Optional) Create a virtual environment:

python -m venv venv

Activate the virtual environment:  
- Linux/macOS: `source venv/bin/activate`  
- Windows: `venv\Scripts\activate`

3. Install dependencies:

pip install -r requirements.txt

4. Update paths in `main.py` if needed:

VIDEO_SOURCE = "video.mp4"  # or your own video file  
TICKET_PRICE = 15  
LINE_POSITION = 300  
MIN_CONFIDENCE = 0.3  
REID_WEIGHTS_PATH = "path_to_your_reid_model.pt"  
TRACKING_CONFIG_PATH = "path_to_botsort_config.yaml"

---

## Usage

Run the script:

python main.py

- Press **q** to quit  
- The annotated output video will be saved as `bus_counter_output.avi`

---

## Requirements

- Python 3.8+  
- OpenCV  
- PyTorch  
- Ultralytics YOLOv8  
- BoxMOT (StrongSORT / BoTSORT)

---

## Future Improvements

- Support live camera feed  
- Dashboard for real-time statistics  
- Multi-bus tracking and analytics

---

## License

MIT License © [Yehia elharery]
