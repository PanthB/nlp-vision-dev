#!/usr/bin/env python3

import sys
import socket
import cv2
import numpy as np
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import time

# Constants
WINDOW_TITLE = "Simple Video Receiver"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
UDP_PORT = 5005
UDP_BIND_ADDRESS = "0.0.0.0"
MAX_UDP_SIZE = 65507  # Maximum UDP packet size
SOCKET_TIMEOUT = 1  # milliseconds
MIN_JPEG_SIZE = 100  # Minimum size for a valid JPEG
JPEG_HEADER = b'\xff\xd8'  # JPEG file header
JPEG_FOOTER = b'\xff\xd9'  # JPEG file footer

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleVideoReceiver(QMainWindow):
    """Simplified video receiver that directly processes and displays frames."""

    def __init__(self) -> None:
        """Initialize the video receiver window and setup components."""
        super().__init__()
        self._setup_window()
        self._setup_ui()
        self._setup_udp_socket()
        self._setup_timer()
        self.packets_received = 0
        self.last_log_time = 0

    def _setup_window(self) -> None:
        """Configure the main window properties."""
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)

    def _setup_ui(self) -> None:
        """Setup the user interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

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
                if current_time - self.last_log_time >= 1.0:
                    logger.info(f"Received {self.packets_received} packets in the last second")
                    self.packets_received = 0
                    self.last_log_time = current_time

                if self._validate_jpeg_data(data):
                    self._process_frame(data)
                else:
                    logger.warning(f"Invalid JPEG data received from {addr}")

        except BlockingIOError:
            pass  # No data available
        except Exception as e:
            logger.error(f"Error in check_socket: {e}")
            self.status_label.setText(f"Error: {str(e)}")

    def _validate_jpeg_data(self, data: bytes) -> bool:
        """Validate if the received data is a valid JPEG."""
        if len(data) < MIN_JPEG_SIZE:
            logger.warning(f"Data too small: {len(data)} bytes")
            return False

        if not data.startswith(JPEG_HEADER):
            logger.warning("Data doesn't start with JPEG header")
            logger.debug(f"First 10 bytes: {data[:10].hex()}")
            return False

        if not data.endswith(JPEG_FOOTER):
            logger.warning("Data doesn't end with JPEG footer")
            logger.debug(f"Last 10 bytes: {data[-10:].hex()}")
            return False

        return True

    def _process_frame(self, jpeg_data: bytes) -> None:
        """Process and display a JPEG frame."""
        try:
            # Convert JPEG data to numpy array
            nparr = np.frombuffer(jpeg_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                self._display_frame(frame)
            else:
                logger.error("Failed to decode JPEG frame")
                logger.debug(f"JPEG data size: {len(jpeg_data)} bytes")

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            logger.debug(f"JPEG data size: {len(jpeg_data)} bytes")

    def _display_frame(self, frame: np.ndarray) -> None:
        """Display the processed frame."""
        height, width, channel = frame.shape

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

        # Scale and display image
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)
        self.status_label.setText(f"Frame size: {width}x{height}")

    def closeEvent(self, event) -> None:
        """Clean up when window is closed."""
        logger.info("Closing receiver...")
        self.timer.stop()
        self.sock.close()
        event.accept()


def main() -> None:
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = SimpleVideoReceiver()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 