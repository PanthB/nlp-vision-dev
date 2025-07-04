#!/usr/bin/env python3
import cv2
import socket
import time
import sys
import os
import struct
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

os.makedirs('logs', exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatters
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_formatter = logging.Formatter('%(levelname)s: %(message)s')

# File handler (with rotation to prevent huge log files)
file_handler = RotatingFileHandler(
    'logs/sender.log',
    maxBytes=1024*1024,  # 1MB
    backupCount=5
)
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("=== Starting new sender session ===")

# UDP Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MAX_UDP_SIZE = 1400  # Safe UDP packet size
HEADER_SIZE = 12  # 4 bytes for frame number, 4 bytes for packet number, 4 bytes for total packets

def send_video(video_path):
    """Send video frames over UDP, simulating FPGA behavior."""
    if not os.path.exists(video_path):
        logger.error(f"Error: Video file not found at {video_path}")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    logger.info(f"Starting video stream from {video_path}")
    logger.info(f"Sending to {UDP_IP}:{UDP_PORT}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video, restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Encode frame to JPEG
            _, encoded = cv2.imencode('.jpg', frame)
            data = encoded.tobytes()
            
            # Calculate number of packets needed for this frame
            total_packets = (len(data) + MAX_UDP_SIZE - HEADER_SIZE - 1) // (MAX_UDP_SIZE - HEADER_SIZE)
            logger.info(f"Frame {frame_number}: Sending {total_packets} packets, total size {len(data)} bytes")
            
            # Split into UDP-sized chunks with headers
            for packet_num in range(total_packets):
                start = packet_num * (MAX_UDP_SIZE - HEADER_SIZE)
                end = min(start + (MAX_UDP_SIZE - HEADER_SIZE), len(data))
                chunk = data[start:end]
                
                # Create header (frame number, packet number, total packets)
                header = struct.pack('>III', frame_number, packet_num, total_packets)
                packet = header + chunk
                
                try:
                    sock.sendto(packet, (UDP_IP, UDP_PORT))
                    if packet_num == 0:  # Log first packet
                        logger.info(f"Sent start of frame {frame_number}, total packets: {total_packets}")
                    elif packet_num == total_packets - 1:  # Log last packet
                        logger.info(f"Sent end of frame {frame_number}, packet {packet_num}/{total_packets-1}")
                except Exception as e:
                    logger.error(f"Error sending packet: {e}")

                time.sleep(0.001)  # Small delay between packets

            frame_number += 1
            # Simulate 30 FPS
            time.sleep(1/30)

    except KeyboardInterrupt:
        logger.info("\nStopping video sender...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        cap.release()
        sock.close()
        logger.info("Sender stopped")

def main():
    video_path = "test_video.mp4"
    logger.info(f"Starting video sender...")
    logger.info(f"Streaming {video_path} to {UDP_IP}:{UDP_PORT}")
    logger.info("Press Ctrl+C to stop")
    
    send_video(video_path)

if __name__ == "__main__":
    main() 