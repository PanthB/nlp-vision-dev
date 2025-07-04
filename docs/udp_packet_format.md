# UDP Packet Format Documentation

## Overview
The video stream receiver expects UDP packets containing JPEG-encoded video frames. Each frame is split into multiple UDP packets to handle the size limitations of UDP. The receiver reassembles these packets into complete frames for display.

## Packet Structure

### Header (12 bytes)
Each UDP packet starts with a 12-byte header containing metadata about the frame and packet:

```
+----------------+----------------+----------------+
| Frame Number   | Packet Number  | Total Packets  |
| (4 bytes)      | (4 bytes)      | (4 bytes)      |
+----------------+----------------+----------------+
```

- **Frame Number** (4 bytes): 
  - Big-endian unsigned integer
  - Increments for each new video frame
  - Used to identify which frame the packet belongs to

- **Packet Number** (4 bytes):
  - Big-endian unsigned integer
  - Zero-based index of the packet within the frame
  - First packet of a frame has packet number 0

- **Total Packets** (4 bytes):
  - Big-endian unsigned integer
  - Total number of packets that make up the complete frame
  - Same for all packets of the same frame

### Payload
Following the header is the actual JPEG data for that portion of the frame:

```
+----------------+----------------+
| Header         | JPEG Data      |
| (12 bytes)     | (variable)     |
+----------------+----------------+
```

## Packet Size
- Maximum UDP packet size: 1400 bytes (MAX_UDP_SIZE)
- Header size: 12 bytes (HEADER_SIZE)
- Maximum payload size: 1388 bytes (1400 - 12)

## Frame Reassembly
1. Receiver maintains a buffer for each frame number
2. Packets are stored in order using their packet number
3. When all packets for a frame are received (packet count matches total_packets), the frame is reassembled
4. The complete JPEG data is then decoded and displayed

## Example
For a frame split into 3 packets:

```
Packet 1:
[Frame: 1][Packet: 0][Total: 3][JPEG data part 1]

Packet 2:
[Frame: 1][Packet: 1][Total: 3][JPEG data part 2]

Packet 3:
[Frame: 1][Packet: 2][Total: 3][JPEG data part 3]
```

## Validation
The receiver performs the following validations:
1. Checks if packet size is at least 12 bytes (header size)
2. Validates JPEG data starts with proper JPEG header (0xFF 0xD8)
3. Ensures minimum JPEG size (100 bytes)
4. Verifies all packets for a frame are received before processing

## Error Handling
- Missing packets are logged with their packet numbers
- Invalid headers are logged with error details
- Corrupted JPEG data is logged with the first few bytes for debugging
- Old frames are cleaned up to prevent memory issues

## Network Configuration
- Default UDP Port: 5005
- Bind Address: 0.0.0.0 (all interfaces)
- Non-blocking socket mode
- Socket check interval: 1 millisecond 