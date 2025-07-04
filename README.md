# Video Stream Simulator with GUI

This project simulates an FPGA-like system sending video frames over UDP, with a GUI interface for displaying the video stream and user input.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the receiver (with GUI) in one terminal:
```bash
python src/receiver.py
```

2. Start the sender in another terminal:
```bash
python src/sender.py
```

## Features

- Simulates video streaming over UDP (like an FPGA would send)
- Displays video stream in a GUI window
- Includes a text input field for user interaction
- Properly handles video frame reassembly from UDP packets

## Project Structure

- `src/sender.py`: Simulates FPGA sending video frames over UDP
- `src/receiver.py`: Receives video frames and displays them in GUI
- `requirements.txt`: Project dependencies
- `README.md`: This file 