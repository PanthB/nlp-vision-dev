#!/usr/bin/env python3

import sys
import socket
import cv2
import numpy as np
import struct
import logging
import time
import traceback
from collections import defaultdict
from typing import Dict, Any

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap


# Constants
WINDOW_TITLE = "Video Stream Receiver"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_X_POS = 100
WINDOW_Y_POS = 100

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
UDP_BIND_ADDRESS = "0.0.0.0"
MAX_UDP_SIZE = 1400  # Safe UDP packet size
HEADER_SIZE = 12  # 4 bytes for frame number, 4 bytes for packet number, 4 bytes for total packets
SOCKET_TIMEOUT = 1  # milliseconds
MIN_JPEG_SIZE = 100
JPEG_HEADER = b'\xff\xd8'
FRAME_BUFFER_SIZE = 5  # Number of frames to keep in buffer

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoReceiver(QMainWindow):
    """Main window class for receiving and displaying video stream."""

    def __init__(self) -> None:
        """Initialize the video receiver window and setup components."""
        super().__init__()
        self._setup_window()
        self._setup_ui()
        self._setup_udp_socket()
        self._setup_frame_buffers()
        self._setup_timer()

    def _setup_window(self) -> None:
        """Configure the main window properties."""
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(WINDOW_X_POS, WINDOW_Y_POS, WINDOW_WIDTH, WINDOW_HEIGHT)

    def _setup_ui(self) -> None:
        """Setup the user interface components."""
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # Text input and submit button
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Filter the video color...")
        layout.addWidget(self.text_input)

        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.handle_submit)
        layout.addWidget(submit_button)

        # Status label
        self.status_label = QLabel("Waiting for video stream...")
        layout.addWidget(self.status_label)

    def _setup_udp_socket(self) -> None:
        """Initialize and configure the UDP socket."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((UDP_BIND_ADDRESS, UDP_PORT))
            logger.info(f"Successfully bound to port {UDP_PORT}")
        except Exception as e:
            logger.error(f"Failed to bind to port {UDP_PORT}: {e}")
            self.status_label.setText(f"Error: Failed to bind to port {UDP_PORT}")
            return

        self.sock.setblocking(False)  # Non-blocking socket

    def _setup_frame_buffers(self) -> None:
        """Initialize frame buffer data structures."""
        self.frame_buffers: Dict[int, Dict[int, bytes]] = defaultdict(dict)
        self.frame_total_packets: Dict[int, int] = {}
        self.current_frame = 0
        self.last_log_time = time.time()
        self.packets_received = 0

    def _setup_timer(self) -> None:
        """Setup the timer for checking UDP socket."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_socket)
        self.timer.start(SOCKET_TIMEOUT)
        logger.info("Receiver initialized and waiting for video stream")

    def check_socket(self) -> None:
        """Check for new UDP packets and process them."""
        try:
            while True:  # Process all available packets
                data, addr = self.sock.recvfrom(MAX_UDP_SIZE)
                self.packets_received += 1

                # Log packet statistics every second
                current_time = time.time()
                # Remove in production
                if current_time - self.last_log_time >= 1.0:
                    logger.info(
                        f"Received {self.packets_received} packets in the last second"
                    )
                    self.packets_received = 0
                    self.last_log_time = current_time

                if len(data) < HEADER_SIZE:
                    logger.warning(f"Received packet too small: {len(data)} bytes")
                    continue

                self._process_packet(data)

        except BlockingIOError:
            pass  # No data available
        except Exception as e:
            logger.error(f"Error in check_socket: {e}")
            self.status_label.setText(f"Error: {str(e)}")

    def _process_packet(self, data: bytes) -> None:
        """Process a single UDP packet."""
        header = data[:HEADER_SIZE]
        try:
            frame_number, packet_num, total_packets = struct.unpack('>III', header)
            packet_data = data[HEADER_SIZE:]

            # Store total packets for this frame
            self.frame_total_packets[frame_number] = total_packets

            # Log first and last packet of each frame
            if packet_num == 0:
                logger.info(
                    f"Received start of frame {frame_number}, "
                    f"expecting {total_packets} packets"
                )
            elif packet_num == total_packets - 1:
                logger.info(
                    f"Received end of frame {frame_number}, "
                    f"packet {packet_num}/{total_packets-1}"
                )

            # Store packet in frame buffer
            self.frame_buffers[frame_number][packet_num] = packet_data

            # Try to process complete frames
            self.process_complete_frames()

        except struct.error as e:
            logger.error(f"Failed to unpack header: {e}, data length: {len(data)}")

    def process_complete_frames(self) -> None:
        """Process any complete frames in the buffer."""
        while self.current_frame in self.frame_buffers:
            frame_packets = self.frame_buffers[self.current_frame]
            total_packets = self.frame_total_packets.get(self.current_frame)

            if total_packets is None:
                logger.warning(f"No total packet count for frame {self.current_frame}")
                break

            if not self._is_frame_complete(frame_packets, total_packets):
                break

            frame_data = self._reassemble_frame(frame_packets, total_packets)
            self.process_frame(frame_data)
            self._cleanup_processed_frame()

    def _is_frame_complete(self, frame_packets: Dict[int, bytes], total_packets: int) -> bool:
        """Check if all packets for a frame have been received."""
        if len(frame_packets) != total_packets:
            missing = set(range(total_packets)) - set(frame_packets.keys())
            logger.info(
                f"Frame {self.current_frame}: have {len(frame_packets)}/{total_packets} "
                f"packets, missing: {missing}"
            )
            return False
        return True

    def _reassemble_frame(self, frame_packets: Dict[int, bytes], total_packets: int) -> bytes:
        """Reassemble frame data from packets."""
        frame_data = b''
        for i in range(total_packets):
            frame_data += frame_packets[i]
        logger.info(
            f"Processing complete frame {self.current_frame} "
            f"({len(frame_data)} bytes)"
        )
        return frame_data

    def _cleanup_processed_frame(self) -> None:
        """Clean up processed frame and old frames."""
        del self.frame_buffers[self.current_frame]
        del self.frame_total_packets[self.current_frame]
        self.current_frame += 1

        # Clean up old frames
        old_frames = [f for f in self.frame_buffers if f < self.current_frame - FRAME_BUFFER_SIZE]
        for f in old_frames:
            logger.debug(f"Cleaning up old frame {f}")
            del self.frame_buffers[f]
            if f in self.frame_total_packets:
                del self.frame_total_packets[f]

    def process_frame(self, jpeg_data: bytes) -> None:
        """Process and display a JPEG frame."""
        try:
            logger.info(f"Starting to process frame, JPEG data size: {len(jpeg_data)} bytes")

            if not self._validate_jpeg_data(jpeg_data):
                return

            frame = self._decode_jpeg_frame(jpeg_data)
            if frame is not None:
                self._display_frame(frame)
            else:
                self._handle_decode_error(jpeg_data)

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.status_label.setText(f"Frame processing error: {str(e)}")

    def _validate_jpeg_data(self, jpeg_data: bytes) -> bool:
        """Validate JPEG data before processing."""
        if len(jpeg_data) < MIN_JPEG_SIZE:
            logger.error(f"Frame data too small: {len(jpeg_data)} bytes")
            return False

        if not jpeg_data.startswith(JPEG_HEADER):
            logger.error("Data doesn't start with JPEG header")
            logger.error(f"First 10 bytes: {jpeg_data[:10]}")
            return False

        return True

    def _decode_jpeg_frame(self, jpeg_data: bytes) -> np.ndarray:
        """Decode JPEG data into a numpy array."""
        nparr = np.frombuffer(jpeg_data, np.uint8)
        logger.info(f"Converted to numpy array of size: {nparr.size}")
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _display_frame(self, frame: np.ndarray) -> None:
        """Display the processed frame."""
        height, width, channel = frame.shape
        logger.info(f"Successfully decoded JPEG to image of size {width}x{height}")

        # Convert frame to QImage
        bytes_per_line = 3 * width
        q_img = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        q_img = q_img.rgbSwapped()  # Convert BGR to RGB
        logger.info("Converted to QImage")

        # Scale and display image
        pixmap = QPixmap.fromImage(q_img)
        if pixmap.isNull():
            logger.error("Failed to create QPixmap from QImage")
            return

        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        logger.info(
            f"Scaled image to {scaled_pixmap.width()}x{scaled_pixmap.height()}"
        )

        self.video_label.setPixmap(scaled_pixmap)
        self.status_label.setText(
            f"Receiving video stream... Frame size: {width}x{height}"
        )
        logger.info("Successfully displayed frame")

    def _handle_decode_error(self, jpeg_data: bytes) -> None:
        """Handle JPEG decode errors."""
        logger.error(
            f"Failed to decode JPEG frame, data size: {len(jpeg_data)} bytes"
        )
        logger.error(f"First 20 bytes: {jpeg_data[:20]}")

    def handle_submit(self) -> None:
        """Handle text input submission."""
        text = self.text_input.text()
        if text:
            print(f"User submitted: {text}")
            self.text_input.clear()
            self.status_label.setText(f"Last submission: {text}")

    def closeEvent(self, event: Any) -> None:
        """Clean up when window is closed."""
        logger.info("Closing receiver...")
        self.timer.stop()
        self.sock.close()
        event.accept()


def main() -> None:
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = VideoReceiver()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 