# Video Stream Receiver with NLP Command Processing

A PyQt6-based video streaming receiver that can process UDP video streams and includes natural language processing for video filter commands.

## Features

- **Video Stream Reception**: Receives JPEG-encoded video frames via UDP
- **Packet Reassembly**: Handles frame reconstruction from multiple UDP packets
- **Resolution Agnostic**: Automatically adapts to any video resolution
- **NLP Command Processing**: Natural language interface for video filter commands
- **Real-time Display**: Low-latency video display with automatic scaling

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies
```bash
pip install PyQt6 opencv-python numpy scikit-learn sentence-transformers
```

### 3. Save Dependencies (Optional)
```bash
pip freeze > requirements.txt
```

## Usage

### Video Receiver
```bash
python src/receiver.py
```
- Receives video streams on UDP port 5005
- Displays video with automatic scaling
- Includes text input for user commands
- Shows received user input in formatted display

### Simple Video Receiver (Alternative)
```bash
python src/new_receiver.py  # If available
```
- Simplified version without packet reassembly
- Assumes complete JPEG frames in single UDP packets

### NLP Command Processor
```bash
python src/nlp_test.py
```
Test natural language commands like:
- "make it grayscale"
- "turn on blur" 
- "make it brighter"
- "disable grayscale"
- "make it darker"

## Project Structure

```
capstone-gui/
├── src/
│   ├── receiver.py          # Main video receiver with NLP input
│   ├── new_receiver.py      # Simplified receiver (if available)
│   └── nlp_test.py         # Standalone NLP command processor
├── docs/
│   └── udp_packet_format.md # UDP packet format documentation
└── README.md
```

## UDP Packet Format

The receiver expects UDP packets with a specific format for video frames. See `docs/udp_packet_format.md` for detailed specifications.

### Header Structure (12 bytes)
- Frame Number (4 bytes)
- Packet Number (4 bytes) 
- Total Packets (4 bytes)

### Payload
- JPEG encoded video data

## Dependencies

- **PyQt6**: GUI framework
- **OpenCV**: Video processing and JPEG decoding
- **NumPy**: Array operations
- **scikit-learn**: Machine learning utilities for NLP
- **sentence-transformers**: Pre-trained language models for semantic similarity

## Configuration

### Video Receiver Constants
- **UDP Port**: 5005 (configurable in source)
- **Packet Size**: 1400 bytes max
- **Window Size**: 800x600 pixels
- **Frame Buffer**: 5 frames max

### NLP Command Processor
- **Model**: all-MiniLM-L6-v2 (lightweight sentence transformer)
- **Similarity Threshold**: 0.6 for command matching
- **Supported Commands**: grayscale, blur, brightness adjustment

## Development

### Adding New NLP Commands
1. Add command phrases to `commands_db` in `nlp_test.py`
2. Define FPGA register mapping in `FPGA_REGISTERS`
3. Test with various natural language inputs

### Video Stream Integration
The NLP processor can be integrated with the video receiver to provide real-time filter control based on user commands.

## Troubleshooting

### Video Issues
- Ensure UDP port 5005 is available
- Check that video data is properly JPEG-encoded
- Verify packet format matches specification

### NLP Issues
- First run may be slower due to model download
- Internet connection required for initial model setup
- Commands must exceed similarity threshold (0.6) to execute 