# Eye Stress Detection System

This project is an AI-powered eye stress detection system that monitors eye movements, blink patterns, and calculates stress levels in real-time using computer vision.

## Features

- Real-time eye tracking and blink detection
- Stress level analysis (Low, Moderate, High)
- Live camera feed with eye landmark visualization
- Event logging and session monitoring
- Final session metrics and analysis
- User-friendly web interface

## Prerequisites

- Python 3.9 or higher
- Webcam
- Windows/Linux/MacOS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CluelessMe/eye_stress_detection1.git
```

2. Create a virtual environment (recommended):
```bash
# Windows
python -m venv eyes
eyes\Scripts\activate

# Linux/MacOS
python -m venv eyes
source eyes/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Activate the virtual environment (if not already activated):
```bash
# Windows
eyes\Scripts\activate

# Linux/MacOS
source eyes/bin/activate
```

2. Run the application:
```bash
cd src
```

```bash - without ai
python app2.py
```

```bash -with ai
python app3.py
```

Note: 
app2.py runs with index1.html
app3.py runs with index.html


## Project Structure

```
perp/
├── src/
│   ├── app.py              # Main application
│   ├── stress_analyzer.py  # Stress analysis logic
│   ├── blink_api_stream.py # API streaming
│   └── blink_detection.py  # Blink detection
├── utils/
│   └── video_processing.py # Video processing utilities
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Troubleshooting

1. **Camera not detected**
   - Ensure your webcam is properly connected
   - Check if other applications are using the camera
   - Try restarting the application

2. **Face not detected**
   - Ensure proper lighting
   - Position yourself directly facing the camera
   - Maintain appropriate distance from camera

3. **Performance issues**
   - Close other resource-intensive applications
   - Ensure your system meets the minimum requirements
   - Try running in a different browser

## System Requirements

- **Processor**: Intel Core i3/AMD Ryzen 3 or better
- **RAM**: 4GB minimum, 8GB recommended
- **Webcam**: 720p or better resolution
- **Internet**: Not required (runs locally)
- **Display**: 1280x720 or higher resolution

## Notes

- For best results, use the application in a well-lit environment
- Take regular breaks to prevent eye strain
- Keep a proper distance from the screen (50-70 cm)
- Ensure stable head position during monitoring






