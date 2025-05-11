# Social Distance Breach Monitoring System

Hey there! This is my final year engineering project (2021) where I built a computer vision system that can detect when people aren't following social distancing rules. It works by analyzing video feeds and measuring distances between people.

## About
I started this project during COVID lockdowns when social distancing was super important. The system uses YOLO to detect people in video and then checks if they're standing too close to each other. It's not perfect, but it works pretty well for a proof-of-concept!

## What it does
- Detects people in real-time using YOLOv3 (which was tricky to implement but works great)
- Figures out how close people are to each other using pixel distances
- Shows violations with red boxes around people who are too close
- Works with both video files and webcam (tested mostly with videos though)
- Runs on Windows/Mac/Linux (I developed on Windows, but it should work everywhere)

## Tech I used
- YOLOv3 for finding people in the frames
- OpenCV for all the video processing
- NumPy for the math stuff and calculations
- Python for everything else

## How to set it up
1. Clone this repo:
```
git clone https://github.com/yourusername/social-distance-breach-detection.git
cd social-distance-breach-detection
```

2. Create a virtual environment (trust me, you want to do this):
```
python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows
```

3. Install all the requirements:
```
pip install -r requirements.txt
```
Note: You might need to install some packages manually if things break.

## Running the code
Basic usage:
```
python run.py --input input/video.mp4 --output output.avi --display 1
```

Command line options:
- `--input`: Your video file (leave empty to use webcam)
- `--output`: Where to save the processed video
- `--display`: Set to 1 to see output in real-time (slows things down a bit)

## Some notes on implementation
- I'm using pixel distances which isn't perfect but works OK for the scope of this project
- You can change the distance threshold in `config/settings.py`:
  ```
  MIN_DISTANCE_PIXELS = 50  # Adjust this based on your video
  ```
- Works best with a fixed camera angle
- Gets around 15-20 FPS on my laptop (i7, 16GB RAM)
- I started working on a more accurate version using perspective transformation but didn't have time to finish it

## Project folders
```
social-distance-breach-detection/
├── input/          # Put your test videos here
├── output/         # Where processed videos get saved
├── yolo-coco/      # YOLO model files (you need to download these separately)
│   ├── yolov3.cfg
│   ├── yolov3.weights  # Download this from darknet site
│   └── coco.names
├── config/         # Settings and configuration
├── utils/          # All the processing code
└── run.py          # Main script to run
```

## Project Description (for my portfolio)
For my final year project, I developed a Social Distancing Monitoring System in response to the COVID-19 pandemic. The system uses computer vision to detect when people aren't maintaining proper distance in public spaces. 

I implemented YOLOv3 for person detection and OpenCV for analyzing distances between detected individuals. The system processes video at 15-20 FPS and highlights social distancing violations with red bounding boxes.

This project taught me a lot about computer vision, object detection, and real-world application of AI concepts. The system could potentially be used in places like stores, train stations, or other public areas where monitoring social distancing is important.

## License
MIT License - feel free to use and modify, just credit me please!
