#!/usr/bin/env python3

import sys
import os
import re

# Must be set before numpy / torch / ctranslate2 load their OpenMP runtimes.
# Without this, macOS aborts when both torch and faster-whisper bundle libiomp5.dylib.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import socket
import time
import cv2
import numpy as np
import struct
from collections import defaultdict
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal, QSize, QRectF, QPointF
from PyQt6.QtGui import QImage, QPixmap, QIcon, QPainter, QPen, QBrush, QColor

from speech_engine import SpeechEngine


def _mic_pixmap(color_hex: str, size: int = 18) -> QPixmap:
    """Return a vector-drawn microphone QPixmap at the requested size."""
    s = float(size)
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)

    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    col   = QColor(color_hex)
    stroke = max(1.5, s * 0.095)

    # ── Capsule body (filled) ─────────────────────────────────────────
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(col))
    bw = s * 0.40
    bh = s * 0.56
    bx = (s - bw) / 2.0
    by = s * 0.03
    p.drawRoundedRect(QRectF(bx, by, bw, bh), bw / 2.0, bw / 2.0)

    # ── U-bracket stand (stroked arc) ─────────────────────────────────
    pen = QPen(col, stroke)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    arc_x = s * 0.11
    arc_y = s * 0.34
    arc_w = s * 0.78
    arc_h = s * 0.44
    # Bottom semicircle: start 0° (3 o'clock), sweep -180° clockwise → 9 o'clock
    p.drawArc(QRectF(arc_x, arc_y, arc_w, arc_h), 0 * 16, -180 * 16)

    # ── Vertical pole ─────────────────────────────────────────────────
    cx       = s / 2.0
    pole_top = arc_y + arc_h
    pole_bot = s * 0.90
    p.drawLine(QPointF(cx, pole_top), QPointF(cx, pole_bot))

    # ── Horizontal base ───────────────────────────────────────────────
    p.drawLine(QPointF(s * 0.25, pole_bot), QPointF(s * 0.75, pole_bot))

    p.end()
    return px


def _mic_icon(color_hex: str, size: int = 18) -> QIcon:
    return QIcon(_mic_pixmap(color_hex, size))


# ── Speech-to-text behaviour ─────────────────────────────────────────
# AUTO_SEND  True  -> submit every transcription automatically
#            False -> populate the input field; user sends manually
AUTO_SEND = False

# ENABLE_JARVIS  True  -> when AUTO_SEND is False, still auto-submit if the
#                         phrase begins with "Jarvis" or "Hey Jarvis";
#                         the wake-word prefix is stripped before sending
#                False -> no wake-word detection; behaves as plain AUTO_SEND=False
ENABLE_JARVIS = True

# Matches "jarvis" or "hey jarvis" (with optional filler punctuation) at the
# very start of a transcription, case-insensitive.
_JARVIS_PREFIX = re.compile(r"^(?:hey\s+)?jarvis[,.\s]*", re.IGNORECASE)

# ── Window ───────────────────────────────────────────────────────────
WINDOW_TITLE  = "NLP-VisionRT"
WINDOW_WIDTH  = 980
WINDOW_HEIGHT = 740
WINDOW_X_POS  = 100
WINDOW_Y_POS  = 150

# ── Video UDP (FPGA → PC) ────────────────────────────────────────────
UDP_IP            = "127.0.0.1"
UDP_PORT          = 5005
UDP_BIND_ADDRESS  = "0.0.0.0"
MAX_UDP_SIZE      = 1400
HEADER_SIZE       = 12
SOCKET_TIMEOUT    = 1
MIN_JPEG_SIZE     = 100
JPEG_HEADER       = b'\xff\xd8'
FRAME_BUFFER_SIZE = 5

# ── Video UDP (FPGA stream_server → PC) ─────────────────────────────
# stream_server.c binds UDP 5013, waits for a client packet, then sends
# fragmented RGB565 frames (same packet format as port 5005).
VIDEO_UDP_PORT = 5013
DEBUG_UDP = True
VIDEO_UDP_START_RETRY_SEC = 3
VIDEO_UDP_WIDTH = 640
VIDEO_UDP_HEIGHT = 480

# ── Command transport (PC → PYNQ) ────────────────────────────────────
# Set COMMAND_TRANSPORT to "ethernet" or "usb" depending on how you
# physically connect the host PC to the PYNQ board.
COMMAND_TRANSPORT = "ethernet"  # "ethernet" | "usb"

# Ethernet / TCP  — PYNQ must run a TCP server on this host:port.
PYNQ_TCP_HOST = "192.168.2.99"
PYNQ_TCP_PORT = 8888

# USB-C / Serial  — PYNQ appears as a CDC serial device on the host.
# macOS example: "/dev/tty.usbmodem1"
# Linux example: "/dev/ttyACM0"
# Windows example: "COM3"
PYNQ_SERIAL_PORT    = "/dev/tty.usbmodem1"
PYNQ_SERIAL_BAUD    = 115200
PYNQ_SERIAL_TIMEOUT = 3.0

# ── Stylesheet ───────────────────────────────────────────────────────
# Light mode: orange / white palette
_ORANGE_50 = "#FFF7ED"
_ORANGE_100 = "#FFEDD5"
_ORANGE_200 = "#FED7AA"
_ORANGE_400 = "#FB923C"
_ORANGE_500 = "#F97316"
_ORANGE_600 = "#EA580C"
_ORANGE_700 = "#C2410C"
_WHITE = "#FFFFFF"
_GRAY_100 = "#F5F5F5"
_GRAY_200 = "#E5E5E5"
_GRAY_400 = "#A3A3A3"
_GRAY_600 = "#525252"
_GRAY_800 = "#262626"
_GRAY_900 = "#171717"

# Dark mode: dark navy + orange palette
_NAVY_50 = "#1e3a5f"
_NAVY_100 = "#172d4d"
_NAVY_200 = "#0f2239"
_NAVY_300 = "#0a1929"
_NAVY_400 = "#061220"
_NAVY_500 = "#030b14"
_ORANGE_DARK = "#F97316"
_ORANGE_DARK_MUTED = "#FB923C"
_SLATE_300 = "#94a3b8"
_SLATE_400 = "#64748b"
_SLATE_500 = "#475569"

APP_STYLESHEET = f"""
QMainWindow {{
    background-color: {_WHITE};
}}
QWidget {{
    background-color: {_WHITE};
    color: {_GRAY_900};
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}}

/* ─── Header ───────────────────────────────────────────────────── */
QWidget#header {{
    background-color: {_ORANGE_50};
    border-bottom: 1px solid {_ORANGE_200};
    min-height: 64px;
    max-height: 64px;
}}
QLabel#logoLabel {{
    color: {_GRAY_900};
    font-size: 17px;
    font-weight: 700;
    background: transparent;
}}
QLabel#logoAccent {{
    color: {_ORANGE_600};
    font-size: 17px;
    font-weight: 700;
    background: transparent;
}}
QLabel#taglineLabel {{
    color: {_GRAY_600};
    font-size: 10px;
    letter-spacing: 0.3px;
    background: transparent;
}}
QLabel#frameCounterLabel {{
    color: {_GRAY_600};
    font-size: 11px;
    font-family: "SF Mono", "Fira Code", monospace;
    padding: 3px 10px;
    background-color: {_ORANGE_100};
    border-radius: 5px;
}}

/* ─── Video container ───────────────────────────────────────────── */
QWidget#videoContainer {{
    background-color: {_GRAY_100};
    border: 1px solid {_ORANGE_200};
    border-radius: 10px;
}}
QWidget#videoTopBar {{
    background-color: {_ORANGE_50};
    border-bottom: 1px solid {_ORANGE_200};
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    min-height: 32px;
    max-height: 32px;
}}
QLabel#liveDot {{
    color: {_ORANGE_600};
    font-size: 10px;
    background: transparent;
}}
QLabel#liveLabel {{
    color: {_ORANGE_600};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.6px;
    background: transparent;
}}
QLabel#resolutionLabel {{
    color: {_GRAY_600};
    font-size: 10px;
    font-family: "SF Mono", "Fira Code", monospace;
    background: transparent;
}}
QLabel#videoLabel {{
    background-color: {_GRAY_100};
    border-bottom-left-radius: 10px;
    border-bottom-right-radius: 10px;
    color: {_GRAY_400};
    font-size: 13px;
}}

/* ─── Command panel ─────────────────────────────────────────────── */
QWidget#commandPanel {{
    background-color: {_ORANGE_50};
    border: 1px solid {_ORANGE_200};
    border-radius: 10px;
}}
QWidget#commandPanelHeader {{
    background-color: {_ORANGE_100};
    border-bottom: 1px solid {_ORANGE_200};
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    min-height: 34px;
    max-height: 34px;
}}
QLabel#commandPanelTitle {{
    color: {_GRAY_600};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    background: transparent;
}}
QLabel#commandIconLabel {{
    color: {_ORANGE_600};
    font-size: 13px;
    background: transparent;
}}
QWidget#responseArea {{
    background: transparent;
}}
QLabel#commandResponseLabel {{
    font-size: 16px;
    background: transparent;
    line-height: 1.6;
}}
QWidget#inputRow {{
    background: transparent;
}}
QLineEdit#commandInput {{
    background-color: {_WHITE};
    border: 1px solid {_ORANGE_200};
    border-radius: 8px;
    color: {_GRAY_900};
    font-size: 13px;
    padding: 9px 13px;
    selection-background-color: {_ORANGE_500};
    selection-color: {_WHITE};
}}
QLineEdit#commandInput:focus {{
    border: 1.5px solid {_ORANGE_500};
}}
QPushButton#submitButton {{
    background-color: {_ORANGE_600};
    color: {_WHITE};
    border: none;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    padding: 9px 20px;
    min-width: 64px;
}}
QPushButton#submitButton:hover {{
    background-color: {_ORANGE_500};
}}
QPushButton#submitButton:pressed {{
    background-color: {_ORANGE_700};
}}
QPushButton#submitButton:disabled {{
    background-color: {_GRAY_200};
    color: {_GRAY_400};
}}

/* ─── Status bar ─────────────────────────────────────────────────── */
QWidget#statusBarWidget {{
    background-color: {_ORANGE_50};
    border-top: 1px solid {_ORANGE_200};
    min-height: 26px;
    max-height: 26px;
}}
QLabel#statusItemLabel {{
    color: {_GRAY_600};
    font-size: 11px;
    font-family: "SF Mono", "Fira Code", monospace;
    background: transparent;
}}

/* ─── Horizontal rule ───────────────────────────────────────────── */
QFrame#hDivider {{
    background-color: {_ORANGE_200};
    max-height: 1px;
    min-height: 1px;
}}

/* ─── Microphone toggle button ──────────────────────────────────── */
QPushButton#micButton {{
    background-color: {_WHITE};
    color: {_GRAY_600};
    border: 1px solid {_ORANGE_200};
    border-radius: 8px;
    font-size: 16px;
    padding: 0px;
    min-width: 38px;
    max-width: 38px;
    min-height: 36px;
    max-height: 36px;
}}
QPushButton#micButton:hover {{
    background-color: {_ORANGE_50};
    color: {_ORANGE_600};
    border-color: {_ORANGE_400};
}}
QPushButton#micButton:disabled {{
    background-color: {_GRAY_100};
    color: {_GRAY_400};
    border-color: {_GRAY_200};
}}

/* ─── Voice-send toggle ─────────────────────────────────────────── */
QPushButton#voiceSendButton {{
    background-color: transparent;
    color: {_GRAY_600};
    border: 1px solid {_ORANGE_200};
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 1px 8px;
}}
QPushButton#voiceSendButton:hover {{
    color: {_ORANGE_600};
    border-color: {_ORANGE_400};
}}
"""

APP_STYLESHEET_DARK = f"""
QMainWindow {{
    background-color: {_NAVY_500};
}}
QWidget {{
    background-color: {_NAVY_500};
    color: {_SLATE_300};
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}}

/* ─── Header ───────────────────────────────────────────────────── */
QWidget#header {{
    background-color: {_NAVY_400};
    border-bottom: 1px solid {_NAVY_50};
    min-height: 64px;
    max-height: 64px;
}}
QLabel#logoLabel {{
    color: {_SLATE_300};
    font-size: 17px;
    font-weight: 700;
    background: transparent;
}}
QLabel#logoAccent {{
    color: {_ORANGE_DARK};
    font-size: 17px;
    font-weight: 700;
    background: transparent;
}}
QLabel#taglineLabel {{
    color: {_SLATE_400};
    font-size: 10px;
    letter-spacing: 0.3px;
    background: transparent;
}}
QLabel#frameCounterLabel {{
    color: {_SLATE_300};
    font-size: 11px;
    font-family: "SF Mono", "Fira Code", monospace;
    padding: 3px 10px;
    background-color: {_NAVY_100};
    border-radius: 5px;
}}

/* ─── Video container ───────────────────────────────────────────── */
QWidget#videoContainer {{
    background-color: {_NAVY_500};
    border: 1px solid {_NAVY_50};
    border-radius: 10px;
}}
QWidget#videoTopBar {{
    background-color: {_NAVY_400};
    border-bottom: 1px solid {_NAVY_50};
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    min-height: 32px;
    max-height: 32px;
}}
QLabel#liveDot {{
    color: {_ORANGE_DARK};
    font-size: 10px;
    background: transparent;
}}
QLabel#liveLabel {{
    color: {_ORANGE_DARK};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.6px;
    background: transparent;
}}
QLabel#resolutionLabel {{
    color: {_SLATE_400};
    font-size: 10px;
    font-family: "SF Mono", "Fira Code", monospace;
    background: transparent;
}}
QLabel#videoLabel {{
    background-color: {_NAVY_500};
    border-bottom-left-radius: 10px;
    border-bottom-right-radius: 10px;
    color: {_SLATE_500};
    font-size: 13px;
}}

/* ─── Command panel ─────────────────────────────────────────────── */
QWidget#commandPanel {{
    background-color: {_NAVY_400};
    border: 1px solid {_NAVY_50};
    border-radius: 10px;
}}
QWidget#commandPanelHeader {{
    background-color: {_NAVY_100};
    border-bottom: 1px solid {_NAVY_50};
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    min-height: 34px;
    max-height: 34px;
}}
QLabel#commandPanelTitle {{
    color: {_SLATE_400};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    background: transparent;
}}
QLabel#commandIconLabel {{
    color: {_ORANGE_DARK};
    font-size: 13px;
    background: transparent;
}}
QWidget#responseArea {{
    background: transparent;
}}
QLabel#commandResponseLabel {{
    font-size: 16px;
    background: transparent;
    line-height: 1.6;
}}
QWidget#inputRow {{
    background: transparent;
}}
QLineEdit#commandInput {{
    background-color: {_NAVY_300};
    border: 1px solid {_NAVY_50};
    border-radius: 8px;
    color: {_SLATE_300};
    font-size: 13px;
    padding: 9px 13px;
    selection-background-color: {_ORANGE_600};
    selection-color: {_WHITE};
}}
QLineEdit#commandInput:focus {{
    border: 1.5px solid {_ORANGE_600};
}}
QPushButton#submitButton {{
    background-color: {_ORANGE_600};
    color: {_WHITE};
    border: none;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    padding: 9px 20px;
    min-width: 64px;
}}
QPushButton#submitButton:hover {{
    background-color: {_ORANGE_500};
}}
QPushButton#submitButton:pressed {{
    background-color: {_ORANGE_700};
}}
QPushButton#submitButton:disabled {{
    background-color: {_NAVY_200};
    color: {_SLATE_500};
}}

/* ─── Status bar ─────────────────────────────────────────────────── */
QWidget#statusBarWidget {{
    background-color: {_NAVY_400};
    border-top: 1px solid {_NAVY_50};
    min-height: 26px;
    max-height: 26px;
}}
QLabel#statusItemLabel {{
    color: {_SLATE_400};
    font-size: 11px;
    font-family: "SF Mono", "Fira Code", monospace;
    background: transparent;
}}

/* ─── Horizontal rule ───────────────────────────────────────────── */
QFrame#hDivider {{
    background-color: {_NAVY_50};
    max-height: 1px;
    min-height: 1px;
}}

/* ─── Microphone toggle button ──────────────────────────────────── */
QPushButton#micButton {{
    background-color: {_NAVY_300};
    color: {_SLATE_400};
    border: 1px solid {_NAVY_50};
    border-radius: 8px;
    font-size: 16px;
    padding: 0px;
    min-width: 38px;
    max-width: 38px;
    min-height: 36px;
    max-height: 36px;
}}
QPushButton#micButton:hover {{
    background-color: {_NAVY_100};
    color: {_ORANGE_DARK};
    border-color: {_ORANGE_600};
}}
QPushButton#micButton:disabled {{
    background-color: {_NAVY_500};
    color: {_SLATE_500};
    border-color: {_NAVY_200};
}}

/* ─── Voice-send toggle ─────────────────────────────────────────── */
QPushButton#voiceSendButton {{
    background-color: transparent;
    color: {_SLATE_400};
    border: 1px solid {_NAVY_50};
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 1px 8px;
}}
QPushButton#voiceSendButton:hover {{
    color: {_ORANGE_DARK};
    border-color: {_ORANGE_600};
}}
"""


# ─────────────────────────────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────────────────────────────

class VideoReceiver(QMainWindow):
    """Main window for NLP-VisionRT — receives and displays the FPGA video stream."""

    def __init__(self) -> None:
        super().__init__()
        self._frame_count = 0
        self._dark_mode = False
        self._setup_window()
        self._setup_ui()
        self._setup_udp_sockets()
        self._setup_frame_buffers()
        self._setup_timer()
        self._setup_speech()

    # ── Window ───────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(WINDOW_X_POS, WINDOW_Y_POS, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMinimumSize(720, 560)
        self.setStyleSheet(APP_STYLESHEET)

    # ── UI construction ──────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_header())

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 12)
        content_layout.setSpacing(12)
        content_layout.addWidget(self._build_video_container(), 1)
        content_layout.addWidget(self._build_command_panel(), 0)
        layout.addWidget(content, 1)

        layout.addWidget(self._build_status_bar())

    # ── Header ───────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setSpacing(0)

        # Single label with inline HTML keeps the three parts pixel-tight
        self.brand_label = QLabel(
            '<span style="color:#171717;font-size:17px;font-weight:700;">NLP-</span>'
            '<span style="color:#EA580C;font-size:17px;font-weight:700;">Vision</span>'
            '<span style="color:#171717;font-size:17px;font-weight:700;">RT</span>'
        )
        self.brand_label.setTextFormat(Qt.TextFormat.RichText)
        self.brand_label.setStyleSheet("background: transparent;")

        tagline = QLabel("Real-Time NLP Video Control  ·  FPGA-Accelerated Pipeline")
        tagline.setObjectName("taglineLabel")

        brand_col = QVBoxLayout()
        brand_col.setSpacing(3)
        brand_col.addWidget(self.brand_label)
        brand_col.addWidget(tagline)

        layout.addLayout(brand_col)
        layout.addStretch(1)

        # Extra breathing room before the right-side cluster
        layout.addSpacing(16)

        self.theme_toggle_btn = QPushButton("☀")
        self.theme_toggle_btn.setObjectName("voiceSendButton")
        self.theme_toggle_btn.setToolTip("Toggle dark mode")
        self.theme_toggle_btn.setFixedSize(36, 28)
        self.theme_toggle_btn.clicked.connect(self._toggle_dark_mode)
        layout.addWidget(self.theme_toggle_btn)

        layout.addSpacing(8)

        self.connection_badge = QLabel("● WAITING FOR STREAM")
        self.connection_badge.setStyleSheet("""
            color: #C2410C;
            background-color: rgba(249, 115, 22, 0.15);
            border: 1px solid rgba(234, 88, 12, 0.40);
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            padding: 4px 14px;
        """)
        layout.addWidget(self.connection_badge)
        layout.addSpacing(20)

        self.frame_counter_label = QLabel("FRAME  0")
        self.frame_counter_label.setObjectName("frameCounterLabel")
        layout.addWidget(self.frame_counter_label)

        return header

    # ── Video container ──────────────────────────────────────────────

    def _build_video_container(self) -> QWidget:
        container = QWidget()
        container.setObjectName("videoContainer")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top_bar = QWidget()
        top_bar.setObjectName("videoTopBar")
        tb_layout = QHBoxLayout(top_bar)
        tb_layout.setContentsMargins(12, 0, 12, 0)
        tb_layout.setSpacing(5)

        live_dot = QLabel("●")
        live_dot.setObjectName("liveDot")
        live_txt = QLabel("LIVE")
        live_txt.setObjectName("liveLabel")
        tb_layout.addWidget(live_dot)
        tb_layout.addWidget(live_txt)
        tb_layout.addStretch(1)

        self.resolution_label = QLabel("No signal")
        self.resolution_label.setObjectName("resolutionLabel")
        tb_layout.addWidget(self.resolution_label)

        layout.addWidget(top_bar)

        self.video_label = QLabel(
            "Waiting for video stream from FPGA\n\nUDP  ·  5005 (JPEG)  ·  5013 (RGB565)"
        )
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.video_label, 1)

        return container

    # ── Command panel ────────────────────────────────────────────────

    def _build_command_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("commandPanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        ph = QWidget()
        ph.setObjectName("commandPanelHeader")
        ph_layout = QHBoxLayout(ph)
        ph_layout.setContentsMargins(14, 0, 14, 0)
        ph_layout.setSpacing(6)

        icon = QLabel("◈")
        icon.setObjectName("commandIconLabel")
        title = QLabel("COMMAND")
        title.setObjectName("commandPanelTitle")
        ph_layout.addWidget(icon)
        ph_layout.addWidget(title)
        ph_layout.addStretch(1)

        self.voice_send_btn = QPushButton("VOICE SEND: OFF")
        self.voice_send_btn.setObjectName("voiceSendButton")
        self.voice_send_btn.setToolTip(
            'When ON, saying "send", "enter", "go", or "submit" auto-sends the command'
        )
        self.voice_send_btn.clicked.connect(self._toggle_voice_send)
        ph_layout.addWidget(self.voice_send_btn)

        layout.addWidget(ph)

        resp_area = QWidget()
        resp_area.setObjectName("responseArea")
        resp_layout = QVBoxLayout(resp_area)
        resp_layout.setContentsMargins(18, 16, 18, 16)

        self.user_input_label = QLabel("No command issued yet.")
        self.user_input_label.setObjectName("commandResponseLabel")
        self.user_input_label.setWordWrap(True)
        self.user_input_label.setStyleSheet(
            "color: #A3A3A3; font-size: 16px; background: transparent;"
        )
        resp_layout.addWidget(self.user_input_label)

        layout.addWidget(resp_area)

        div = QFrame()
        div.setObjectName("hDivider")
        div.setFrameShape(QFrame.Shape.HLine)
        div.setFixedHeight(1)
        layout.addWidget(div)

        input_row = QWidget()
        input_row.setObjectName("inputRow")
        ir_layout = QHBoxLayout(input_row)
        ir_layout.setContentsMargins(12, 10, 12, 12)
        ir_layout.setSpacing(8)

        self.text_input = QLineEdit()
        self.text_input.setObjectName("commandInput")
        self.text_input.setPlaceholderText(
            'e.g. "track red"  or  "zoom in"'
        )
        self.text_input.returnPressed.connect(self.handle_submit)
        ir_layout.addWidget(self.text_input, 1)

        self.mic_button = QPushButton()
        self.mic_button.setObjectName("micButton")
        self.mic_button.setIcon(_mic_icon("#525252", 18))
        self.mic_button.setIconSize(QSize(18, 18))
        self.mic_button.setToolTip("Speech model loading…")
        self.mic_button.setEnabled(False)
        self.mic_button.clicked.connect(self._toggle_mic)
        ir_layout.addWidget(self.mic_button)

        self.submit_button = QPushButton("Send")
        self.submit_button.setObjectName("submitButton")
        self.submit_button.clicked.connect(self.handle_submit)
        ir_layout.addWidget(self.submit_button)

        layout.addWidget(input_row)

        return panel

    # ── Status bar ───────────────────────────────────────────────────

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("statusBarWidget")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(0)

        self.status_label = QLabel(f"⬤  UDP:{UDP_PORT}  ·  {VIDEO_UDP_PORT}")
        self.status_label.setObjectName("statusItemLabel")
        self.status_label.setStyleSheet(
            "color: #EA580C; font-size: 11px;"
            " font-family: 'SF Mono','Fira Code',monospace; background: transparent;"
        )
        layout.addWidget(self.status_label)

        layout.addWidget(self._sep())

        self.frames_label = QLabel("Frames: 0")
        self.frames_label.setObjectName("statusItemLabel")
        layout.addWidget(self.frames_label)

        layout.addWidget(self._sep())

        # PYNQ connection config label
        if COMMAND_TRANSPORT == "ethernet":
            pynq_cfg_text = f"PYNQ  {PYNQ_TCP_HOST}:{PYNQ_TCP_PORT}"
        else:
            pynq_cfg_text = f"PYNQ  {PYNQ_SERIAL_PORT}"

        pynq_cfg = QLabel(pynq_cfg_text)
        pynq_cfg.setObjectName("statusItemLabel")
        layout.addWidget(pynq_cfg)

        layout.addWidget(self._sep())

        # TX status — updates on each command send attempt
        self.pynq_status_label = QLabel("TX  —")
        self.pynq_status_label.setObjectName("statusItemLabel")
        layout.addWidget(self.pynq_status_label)

        layout.addStretch(1)

        ver = QLabel("NLP-VisionRT  v1.0")
        ver.setObjectName("statusItemLabel")
        layout.addWidget(ver)

        return bar

    def _sep(self) -> QLabel:
        s = QLabel("  |  ")
        s.setObjectName("statusItemLabel")
        return s

    # ── Theme (dark mode) ─────────────────────────────────────────────

    def _toggle_dark_mode(self) -> None:
        self._dark_mode = not self._dark_mode
        self.setStyleSheet(APP_STYLESHEET_DARK if self._dark_mode else APP_STYLESHEET)
        self.theme_toggle_btn.setText("🌙" if self._dark_mode else "☀")
        self.theme_toggle_btn.setToolTip("Switch to light mode" if self._dark_mode else "Switch to dark mode")
        self._apply_theme_dependent_styles()

    def _apply_theme_dependent_styles(self) -> None:
        """Refresh inline styles for widgets that override the main stylesheet."""
        dm = self._dark_mode
        # Brand label
        if dm:
            self.brand_label.setText(
                '<span style="color:#94a3b8;font-size:17px;font-weight:700;">NLP-</span>'
                '<span style="color:#F97316;font-size:17px;font-weight:700;">Vision</span>'
                '<span style="color:#94a3b8;font-size:17px;font-weight:700;">RT</span>'
            )
        else:
            self.brand_label.setText(
                '<span style="color:#171717;font-size:17px;font-weight:700;">NLP-</span>'
                '<span style="color:#EA580C;font-size:17px;font-weight:700;">Vision</span>'
                '<span style="color:#171717;font-size:17px;font-weight:700;">RT</span>'
            )
        # Connection badge
        if dm:
            self.connection_badge.setStyleSheet("""
                color: #F97316;
                background-color: rgba(249, 115, 22, 0.2);
                border: 1px solid rgba(249, 115, 22, 0.5);
                border-radius: 10px;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 14px;
            """)
        else:
            self.connection_badge.setStyleSheet("""
                color: #C2410C;
                background-color: rgba(249, 115, 22, 0.15);
                border: 1px solid rgba(234, 88, 12, 0.40);
                border-radius: 10px;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 14px;
            """)
        # Status label
        st = self.status_label.text()
        if "⚠" in st or "Failed" in st:
            css = "color: #DC2626;" if not dm else "color: #f87171;"
        elif "⬤" in st:
            css = "color: #EA580C;" if not dm else "color: #F97316;"
        else:
            css = "color: #525252;" if not dm else "color: #64748b;"
        self.status_label.setStyleSheet(
            f"{css} font-size: 11px; font-family: 'SF Mono','Fira Code',monospace; background: transparent;"
        )
        # PYNQ status label
        pt = self.pynq_status_label.text()
        if "✓" in pt:
            pynq_css = "color: #EA580C;" if not dm else "color: #F97316;"
        elif "✗" in pt:
            pynq_css = "color: #DC2626;" if not dm else "color: #f87171;"
        else:
            pynq_css = "color: #525252;" if not dm else "color: #64748b;"
        self.pynq_status_label.setStyleSheet(
            f"{pynq_css} font-size: 11px; font-family: 'SF Mono','Fira Code',monospace; background: transparent;"
        )
        # User input label
        ut = self.user_input_label.text()
        if "Listening" in ut or "Detecting" in ut:
            u_css = "color: #EA580C;" if not dm else "color: #F97316;"
        elif "Speech error" in ut:
            u_css = "color: #DC2626;" if not dm else "color: #f87171;"
        elif "No command" in ut or "ready" in ut.lower() or "Loading" in ut:
            u_css = "color: #A3A3A3;" if not dm else "color: #64748b;"
        else:
            u_css = "color: #525252;" if not dm else "color: #94a3b8;"
        self.user_input_label.setStyleSheet(f"{u_css} font-size: 16px; background: transparent;")
        # Voice send button (ON state)
        if hasattr(self, "_speech_engine") and self._speech_engine.voice_send_enabled:
            if dm:
                self.voice_send_btn.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(249, 115, 22, 0.2);
                        border: 1px solid rgba(249, 115, 22, 0.5);
                        border-radius: 4px;
                        color: #F97316;
                        font-size: 10px;
                        font-weight: 600;
                        letter-spacing: 0.5px;
                        padding: 1px 8px;
                    }
                """)
            else:
                self.voice_send_btn.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(249, 115, 22, 0.12);
                        border: 1px solid rgba(234, 88, 12, 0.40);
                        border-radius: 4px;
                        color: #EA580C;
                        font-size: 10px;
                        font-weight: 600;
                        letter-spacing: 0.5px;
                        padding: 1px 8px;
                    }
                """)
        else:
            self.voice_send_btn.setStyleSheet("")
        # Mic button (listening state)
        if hasattr(self, "_speech_engine") and self._speech_engine.is_listening:
            if dm:
                self.mic_button.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(249, 115, 22, 0.2);
                        border: 1.5px solid #F97316;
                        border-radius: 8px;
                        padding: 0px;
                        min-width: 38px;
                        max-width: 38px;
                        min-height: 36px;
                        max-height: 36px;
                    }
                """)
            else:
                self.mic_button.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(249, 115, 22, 0.15);
                        border: 1.5px solid #EA580C;
                        border-radius: 8px;
                        padding: 0px;
                        min-width: 38px;
                        max-width: 38px;
                        min-height: 36px;
                        max-height: 36px;
                    }
                """)
        else:
            self.mic_button.setStyleSheet("")

    # ── Socket / network setup ───────────────────────────────────────

    def _setup_udp_sockets(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((UDP_BIND_ADDRESS, UDP_PORT))
        except Exception:
            self.status_label.setText(f"⚠  Failed to bind UDP:{UDP_PORT}")
            self.status_label.setStyleSheet(
                "color: #DC2626; font-size: 11px;"
                " font-family: 'SF Mono','Fira Code',monospace; background: transparent;"
            )
            self._apply_theme_dependent_styles()
            return
        self.sock.setblocking(False)

        self.sock_5013 = None
        self._last_start_sent = 0.0
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((UDP_BIND_ADDRESS, VIDEO_UDP_PORT))
            s.setblocking(False)
            if DEBUG_UDP:
                print(f"[receiver] Sending START to {PYNQ_TCP_HOST}:{VIDEO_UDP_PORT}")
            s.sendto(b"START", (PYNQ_TCP_HOST, VIDEO_UDP_PORT))
            self.sock_5013 = s
            self._last_start_sent = time.monotonic()
            if DEBUG_UDP:
                print(f"[receiver] Bound UDP 5005 and 5013. Listening for video.")
        except Exception as exc:
            if DEBUG_UDP:
                print(f"[receiver] Failed to bind 5013: {exc}")
            self.status_label.setText(f"⚠  Failed to bind UDP:{VIDEO_UDP_PORT}")
            self.status_label.setStyleSheet(
                "color: #DC2626; font-size: 11px;"
                " font-family: 'SF Mono','Fira Code',monospace; background: transparent;"
            )
            self._apply_theme_dependent_styles()

    def _setup_frame_buffers(self) -> None:
        self.frame_buffers: Dict[int, Dict[int, bytes]] = defaultdict(dict)
        self.frame_total_packets: Dict[int, int] = {}
        self.current_frame = 0
        self.frame_buffers_5013: Dict[int, Dict[int, bytes]] = defaultdict(dict)
        self.frame_total_packets_5013: Dict[int, int] = {}
        self.current_frame_5013 = 0
        self._rgb565_connected = False

    def _setup_timer(self) -> None:
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_socket)
        self.timer.start(SOCKET_TIMEOUT)

    # ── Socket processing ────────────────────────────────────────────

    def check_socket(self) -> None:
        try:
            while True:
                data, _addr = self.sock.recvfrom(MAX_UDP_SIZE)
                if len(data) >= HEADER_SIZE:
                    self._process_packet(data)
        except BlockingIOError:
            pass
        except Exception as e:
            self.status_label.setText(f"⚠  {str(e)}")
            self.status_label.setStyleSheet(
                "color: #DC2626; font-size: 11px;"
                " font-family: 'SF Mono','Fira Code',monospace; background: transparent;"
            )
            self._apply_theme_dependent_styles()

        if self.sock_5013 is None:
            return
        try:
            while True:
                data, _addr = self.sock_5013.recvfrom(MAX_UDP_SIZE)
                if not self._rgb565_connected:
                    self._rgb565_connected = True
                    if DEBUG_UDP:
                        print("[receiver] First RGB565 packet received.")
                if len(data) >= HEADER_SIZE:
                    self._process_packet_5013(data)
        except BlockingIOError:
            pass
        except Exception as e:
            if DEBUG_UDP:
                print(f"[receiver] sock_5013 error: {e}")

        if not self._rgb565_connected and self.sock_5013 is not None:
            now = time.monotonic()
            if now - self._last_start_sent >= VIDEO_UDP_START_RETRY_SEC:
                try:
                    self.sock_5013.sendto(b"START", (PYNQ_TCP_HOST, VIDEO_UDP_PORT))
                    self._last_start_sent = now
                    if DEBUG_UDP:
                        print(f"[receiver] Resent START to {PYNQ_TCP_HOST}:{VIDEO_UDP_PORT}")
                except Exception:
                    pass

    def _process_packet(self, data: bytes) -> None:
        header = data[:HEADER_SIZE]
        try:
            frame_number, packet_num, total_packets = struct.unpack('>III', header)
            self.frame_total_packets[frame_number] = total_packets
            self.frame_buffers[frame_number][packet_num] = data[HEADER_SIZE:]
            self.process_complete_frames()
        except struct.error:
            pass

    def process_complete_frames(self) -> None:
        while self.current_frame in self.frame_buffers:
            frame_packets = self.frame_buffers[self.current_frame]
            total_packets = self.frame_total_packets.get(self.current_frame)

            if total_packets is None:
                break

            if not self._is_frame_complete(frame_packets, total_packets):
                break

            frame_data = self._reassemble_frame(frame_packets, total_packets)
            self.process_frame(frame_data)
            self._cleanup_processed_frame()

    def _is_frame_complete(
        self, frame_packets: Dict[int, bytes], total_packets: int
    ) -> bool:
        return len(frame_packets) == total_packets

    def _reassemble_frame(
        self, frame_packets: Dict[int, bytes], total_packets: int
    ) -> bytes:
        return b''.join(frame_packets[i] for i in range(total_packets))

    def _cleanup_processed_frame(self) -> None:
        del self.frame_buffers[self.current_frame]
        del self.frame_total_packets[self.current_frame]
        self.current_frame += 1

        old_frames = [
            f for f in self.frame_buffers if f < self.current_frame - FRAME_BUFFER_SIZE
        ]
        for f in old_frames:
            del self.frame_buffers[f]
            if f in self.frame_total_packets:
                del self.frame_total_packets[f]

    def _process_packet_5013(self, data: bytes) -> None:
        header = data[:HEADER_SIZE]
        try:
            frame_number, packet_num, total_packets = struct.unpack(">III", header)
            self.frame_total_packets_5013[frame_number] = total_packets
            self.frame_buffers_5013[frame_number][packet_num] = data[HEADER_SIZE:]
            self.process_complete_frames_5013()
        except struct.error:
            pass

    def process_complete_frames_5013(self) -> None:
        while True:
            next_complete = self._find_next_complete_frame_5013()
            if next_complete is None:
                break
            self._skip_to_frame_5013(next_complete)
            frame_packets = self.frame_buffers_5013[next_complete]
            total_packets = self.frame_total_packets_5013[next_complete]
            frame_data = self._reassemble_frame(frame_packets, total_packets)
            self.process_frame_rgb565(frame_data)
            self._cleanup_processed_frame_5013(next_complete)

    def _find_next_complete_frame_5013(self) -> Optional[int]:
        if self.current_frame_5013 in self.frame_buffers_5013:
            fp = self.frame_buffers_5013[self.current_frame_5013]
            tp = self.frame_total_packets_5013.get(self.current_frame_5013)
            if tp is not None and self._is_frame_complete(fp, tp):
                return self.current_frame_5013
        for fn in sorted(self.frame_buffers_5013.keys()):
            if fn < self.current_frame_5013:
                continue
            fp = self.frame_buffers_5013[fn]
            tp = self.frame_total_packets_5013.get(fn)
            if tp is not None and self._is_frame_complete(fp, tp):
                return fn
        return None

    def _skip_to_frame_5013(self, target: int) -> None:
        for fn in range(self.current_frame_5013, target):
            if fn in self.frame_buffers_5013:
                del self.frame_buffers_5013[fn]
            if fn in self.frame_total_packets_5013:
                del self.frame_total_packets_5013[fn]
        self.current_frame_5013 = target

    def _cleanup_processed_frame_5013(self, frame_num: int) -> None:
        del self.frame_buffers_5013[frame_num]
        del self.frame_total_packets_5013[frame_num]
        self.current_frame_5013 = frame_num + 1
        old = [
            f
            for f in self.frame_buffers_5013
            if f < self.current_frame_5013 - FRAME_BUFFER_SIZE
        ]
        for f in old:
            del self.frame_buffers_5013[f]
            if f in self.frame_total_packets_5013:
                del self.frame_total_packets_5013[f]

    def process_frame_rgb565(self, raw: bytes) -> None:
        try:
            expected = VIDEO_UDP_WIDTH * VIDEO_UDP_HEIGHT * 2
            if len(raw) != expected:
                return
            pixels = np.frombuffer(raw, dtype="<u2").reshape(
                VIDEO_UDP_HEIGHT, VIDEO_UDP_WIDTH
            )
            r = ((pixels >> 11) & 0x1F) * 255 // 31
            g = ((pixels >> 5) & 0x3F) * 255 // 63
            b = (pixels & 0x1F) * 255 // 31
            frame = np.ascontiguousarray(
                np.stack(
                    [b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)],
                    axis=2,
                )
            )
            self._display_frame(frame)
        except Exception as e:
            if DEBUG_UDP:
                print(f"[receiver] RGB565 decode error: {e}")

    # ── Frame decoding / display ─────────────────────────────────────

    def process_frame(self, jpeg_data: bytes) -> None:
        try:
            if not self._validate_jpeg_data(jpeg_data):
                return

            frame = self._decode_jpeg_frame(jpeg_data)
            if frame is not None:
                self._display_frame(frame)

        except Exception as e:
            self.status_label.setText(f"⚠  Frame error: {str(e)}")
            self.status_label.setStyleSheet(
                "color: #DC2626; font-size: 11px;"
                " font-family: 'SF Mono','Fira Code',monospace; background: transparent;"
            )
            self._apply_theme_dependent_styles()

    def _validate_jpeg_data(self, jpeg_data: bytes) -> bool:
        return len(jpeg_data) >= MIN_JPEG_SIZE and jpeg_data.startswith(JPEG_HEADER)

    def _decode_jpeg_frame(self, jpeg_data: bytes) -> np.ndarray:
        nparr = np.frombuffer(jpeg_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _display_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return
        bytes_per_line = 3 * width
        q_img = QImage(
            frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )
        q_img = q_img.rgbSwapped()

        pixmap = QPixmap.fromImage(q_img)
        if pixmap.isNull():
            return

        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.video_label.setText("")
        self.video_label.setPixmap(scaled_pixmap)

        self._frame_count += 1
        self.resolution_label.setText(f"{width}×{height}  ·  RGB888")
        self.frame_counter_label.setText(f"FRAME  {self._frame_count:,}")
        self.frames_label.setText(f"Frames: {self._frame_count:,}")

        self.connection_badge.setText("● LIVE")
        self.connection_badge.setStyleSheet("""
            color: #C2410C;
            background-color: rgba(249, 115, 22, 0.15);
            border: 1px solid rgba(234, 88, 12, 0.40);
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            padding: 4px 14px;
        """)
        self.status_label.setText(f"⬤  UDP:{UDP_PORT}  ·  {VIDEO_UDP_PORT}")
        self.status_label.setStyleSheet(
            "color: #EA580C; font-size: 11px;"
            " font-family: 'SF Mono','Fira Code',monospace; background: transparent;"
        )
        self._apply_theme_dependent_styles()

    # ── Command handling ─────────────────────────────────────────────

    def handle_submit(self) -> None:
        text = self.text_input.text().strip()
        if not text:
            self.user_input_label.setStyleSheet(
                "color: #484F58; font-size: 16px; background: transparent;"
            )
            self.user_input_label.setText("No command issued yet.")
            self._apply_theme_dependent_styles()
            return

        self.text_input.clear()
        self.user_input_label.setStyleSheet(
            "color: #525252; font-size: 16px; background: transparent;"
        )
        self.user_input_label.setText(f'Sent: "{text}"')
        self._apply_theme_dependent_styles()

        self._send_command_to_pynq(text)

    def _send_command_to_pynq(self, text: str) -> None:
        """Dispatch raw command string to the PYNQ board in a background thread."""
        self._cmd_thread = QThread()
        self._cmd_worker = CommandSendWorker(text)
        self._cmd_worker.moveToThread(self._cmd_thread)
        self._cmd_thread.started.connect(self._cmd_worker.run)
        self._cmd_worker.success.connect(self._on_cmd_sent)
        self._cmd_worker.error.connect(self._on_cmd_error)
        self._cmd_worker.success.connect(self._cmd_thread.quit)
        self._cmd_worker.success.connect(self._cmd_worker.deleteLater)
        self._cmd_worker.error.connect(self._cmd_thread.quit)
        self._cmd_worker.error.connect(self._cmd_worker.deleteLater)
        self._cmd_thread.finished.connect(self._cmd_thread.deleteLater)
        self._cmd_thread.start()

    def _on_cmd_sent(self, _text: str) -> None:
        self.pynq_status_label.setText("TX  ✓")
        self.pynq_status_label.setStyleSheet(
            "color: #EA580C; font-size: 11px;"
            " font-family: 'SF Mono','Fira Code',monospace; background: transparent;"
        )
        self._apply_theme_dependent_styles()
        QTimer.singleShot(3000, self._reset_pynq_status)

    def _on_cmd_error(self, _error: str) -> None:
        self.pynq_status_label.setText("TX  ✗")
        self.pynq_status_label.setStyleSheet(
            "color: #DC2626; font-size: 11px;"
            " font-family: 'SF Mono','Fira Code',monospace; background: transparent;"
        )
        self._apply_theme_dependent_styles()
        QTimer.singleShot(5000, self._reset_pynq_status)

    def _reset_pynq_status(self) -> None:
        self.pynq_status_label.setText("TX  —")
        self.pynq_status_label.setStyleSheet(
            "color: #525252; font-size: 11px;"
            " font-family: 'SF Mono','Fira Code',monospace; background: transparent;"
        )
        self._apply_theme_dependent_styles()

    # ── Speech-to-text ────────────────────────────────────────────────

    def _setup_speech(self) -> None:
        self._speech_engine = SpeechEngine(parent=self)
        self._speech_engine.transcript_partial.connect(self._on_transcript_partial)
        self._speech_engine.transcript_final.connect(self._on_transcript_final)
        self._speech_engine.voice_send_triggered.connect(self._on_voice_send_triggered)
        self._speech_engine.listening_started.connect(self._on_listening_started)
        self._speech_engine.listening_stopped.connect(self._on_listening_stopped)
        self._speech_engine.model_status.connect(self._on_speech_model_status)
        self._speech_engine.error.connect(self._on_speech_error)

    def _toggle_mic(self) -> None:
        self._speech_engine.toggle_listening()

    def _toggle_voice_send(self) -> None:
        enabled = not self._speech_engine.voice_send_enabled
        self._speech_engine.voice_send_enabled = enabled
        if enabled:
            self.voice_send_btn.setText("VOICE SEND: ON")
            self.voice_send_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(249, 115, 22, 0.12);
                    border: 1px solid rgba(234, 88, 12, 0.40);
                    border-radius: 4px;
                    color: #EA580C;
                    font-size: 10px;
                    font-weight: 600;
                    letter-spacing: 0.5px;
                    padding: 1px 8px;
                }
            """)
        else:
            self.voice_send_btn.setText("VOICE SEND: OFF")
            self.voice_send_btn.setStyleSheet("")
        self._apply_theme_dependent_styles()

    def _on_listening_started(self) -> None:
        self.mic_button.setIcon(_mic_icon("#EA580C", 18))
        self.mic_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(249, 115, 22, 0.15);
                border: 1.5px solid #EA580C;
                border-radius: 8px;
                padding: 0px;
                min-width: 38px;
                max-width: 38px;
                min-height: 36px;
                max-height: 36px;
            }
        """)
        self.user_input_label.setStyleSheet(
            "color: #EA580C; font-size: 16px; background: transparent;"
        )
        self.user_input_label.setText("Listening…")
        self._apply_theme_dependent_styles()

    def _on_listening_stopped(self) -> None:
        self.mic_button.setIcon(_mic_icon("#525252", 18))
        self.mic_button.setStyleSheet("")
        status_text = self.user_input_label.text()
        if status_text in ("Listening…", "Detecting speech…"):
            self.user_input_label.setStyleSheet(
                "color: #A3A3A3; font-size: 16px; background: transparent;"
            )
            self.user_input_label.setText("No command issued yet.")
        self._apply_theme_dependent_styles()

    def _on_transcript_partial(self, text: str) -> None:
        self.user_input_label.setStyleSheet(
            "color: #525252; font-size: 16px; background: transparent;"
        )
        self.user_input_label.setText(text)
        self._apply_theme_dependent_styles()

    def _on_transcript_final(self, text: str) -> None:
        if AUTO_SEND:
            self.text_input.setText(text)
            self.handle_submit()
            return

        if ENABLE_JARVIS:
            match = _JARVIS_PREFIX.match(text)
            if match:
                command = text[match.end():].strip()
                self.text_input.setText(command)
                self.handle_submit()
                return

        self.text_input.setText(text)

    def _on_voice_send_triggered(self) -> None:
        self.handle_submit()

    def _on_speech_model_status(self, msg: str) -> None:
        self.user_input_label.setStyleSheet(
            "color: #525252; font-size: 16px; background: transparent;"
        )
        self.user_input_label.setText(msg)
        if "ready" in msg.lower():
            self.mic_button.setEnabled(True)
            self.mic_button.setIcon(_mic_icon("#525252", 18))
            self.mic_button.setToolTip("Click to start voice input")
        self._apply_theme_dependent_styles()

    def _on_speech_error(self, msg: str) -> None:
        self.user_input_label.setStyleSheet(
            "color: #DC2626; font-size: 16px; background: transparent;"
        )
        self.user_input_label.setText(f"Speech error: {msg}")
        self._apply_theme_dependent_styles()

    # ── Lifecycle ────────────────────────────────────────────────────

    def closeEvent(self, event: Any) -> None:
        self.timer.stop()
        self.sock.close()
        if self.sock_5013 is not None:
            self.sock_5013.close()
        if hasattr(self, "_speech_engine"):
            self._speech_engine.shutdown()
        event.accept()


# ─────────────────────────────────────────────────────────────────────
#  Command Send Worker  —  transmits raw string to PYNQ board
# ─────────────────────────────────────────────────────────────────────

class CommandSendWorker(QObject):
    """
    Sends a cleaned command string to the PYNQ board.

    Transport selection is controlled by COMMAND_TRANSPORT at the top of this file.

    Ethernet / TCP
    ──────────────
    The PYNQ board must be running a TCP server that accepts a newline-terminated
    UTF-8 string on PYNQ_TCP_HOST:PYNQ_TCP_PORT.

    Minimal PYNQ server example (Python, run on the board):
        import socket
        srv = socket.socket()
        srv.bind(("0.0.0.0", 8888))
        srv.listen(1)
        while True:
            conn, _ = srv.accept()
            cmd = conn.recv(1024).decode().strip()
            print("Received:", cmd)
            conn.close()

    USB-C / Serial
    ──────────────
    The PYNQ board must read from its USB CDC serial port (ttyACM0 / ttyUSB0).
    Requires pyserial on the host: pip install pyserial

    Minimal PYNQ serial reader example (Python, run on the board):
        import sys
        for line in sys.stdin:
            cmd = line.strip()
            print("Received:", cmd)
    """
    success = pyqtSignal(str)
    error   = pyqtSignal(str)

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def run(self) -> None:
        try:
            if COMMAND_TRANSPORT == "ethernet":
                self._send_udp()
            elif COMMAND_TRANSPORT == "usb":
                self._send_serial()
            else:
                raise ValueError(f"Unknown COMMAND_TRANSPORT: '{COMMAND_TRANSPORT}'")
            self.success.emit(self.text)
        except Exception as e:
            self.error.emit(str(e))

    def _send_udp(self) -> None:
        """Send the command as a single UDP datagram to the PYNQ board."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto((self.text + "\n").encode("utf-8"), (PYNQ_TCP_HOST, PYNQ_TCP_PORT))

    def _send_serial(self) -> None:
        """Open the CDC serial port to the PYNQ and write the command, newline-terminated."""
        import serial  # pip install pyserial
        with serial.Serial(
            PYNQ_SERIAL_PORT,
            PYNQ_SERIAL_BAUD,
            timeout=PYNQ_SERIAL_TIMEOUT,
        ) as ser:
            ser.write((self.text + "\n").encode("utf-8"))


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    window = VideoReceiver()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
