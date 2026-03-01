#!/usr/bin/env python3
import cv2
import socket
import time
import sys
import os
import struct
import logging
from logging.handlers import RotatingFileHandler

os.makedirs('logs', exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

file_handler = RotatingFileHandler('logs/sender.log', maxBytes=1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
file_handler.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
console_handler.setLevel(logging.WARNING)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

UDP_IP        = "127.0.0.1"
UDP_PORT      = 5005
MAX_UDP_SIZE  = 1400
HEADER_SIZE   = 12


def send_video(video_path):
    """Send video frames over UDP, simulating FPGA behavior."""
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap  = cv2.VideoCapture(video_path)
    frame_number = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            _, encoded = cv2.imencode('.jpg', frame)
            data = encoded.tobytes()

            total_packets = (
                (len(data) + MAX_UDP_SIZE - HEADER_SIZE - 1) // (MAX_UDP_SIZE - HEADER_SIZE)
            )

            for packet_num in range(total_packets):
                start  = packet_num * (MAX_UDP_SIZE - HEADER_SIZE)
                end    = min(start + (MAX_UDP_SIZE - HEADER_SIZE), len(data))
                chunk  = data[start:end]
                header = struct.pack('>III', frame_number, packet_num, total_packets)

                try:
                    sock.sendto(header + chunk, (UDP_IP, UDP_PORT))
                except Exception as e:
                    logger.error(f"Error sending packet: {e}")

                time.sleep(0.001)

            frame_number += 1
            time.sleep(1 / 60)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        cap.release()
        sock.close()


def main():
    video_path = "test_video.mp4"
    send_video(video_path)


if __name__ == "__main__":
    main()
