#!/usr/bin/env python3
"""
speech_engine.py — Real-time speech-to-text for NLP-VisionRT.

Backed by faster-whisper (OpenAI Whisper via CTranslate2) for Whisper-quality
accuracy at 2-4× the speed of the original implementation. Fully offline,
no API key required. Model is auto-downloaded on first use.

Model options (set MODEL_SIZE below):
    "tiny.en"   ~40 MB   — fastest, ~0.3 s/command on i9 CPU
    "base.en"   ~145 MB  — recommended, ~0.8 s/command on i9 CPU  ← default
    "small.en"  ~470 MB  — high accuracy, ~2 s/command on i9 CPU

Required dependencies:
    pip install faster-whisper sounddevice numpy

Signals
-------
transcript_partial   : str  — status while recording ("Detecting speech…" etc.)
transcript_final     : str  — committed transcription text
voice_send_triggered :      — send-keyword detected (voice-send mode only)
listening_started    :      — microphone opened
listening_stopped    :      — microphone closed
model_status         : str  — loading / ready message
error                : str  — any runtime error
"""

from __future__ import annotations

import queue
import threading
from typing import List

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

# ── Model configuration ───────────────────────────────────────────────
MODEL_SIZE   = "tiny.en"   # "tiny.en" | "base.en" | "small.en"
COMPUTE_TYPE = "int8"      # fastest on CPU; change to "float32" if problems

# ── Audio / VAD settings ──────────────────────────────────────────────
SAMPLE_RATE      = 16_000                              # Hz — Whisper native
CHUNK_DURATION_S = 0.1                                 # seconds per chunk
BLOCK_SIZE       = int(SAMPLE_RATE * CHUNK_DURATION_S) # 1600 samples
SPEECH_TIMEOUT_S = 1.2     # silence gap that triggers transcription
MIN_SPEECH_S     = 0.4     # ignore clips shorter than this

_SILENCE_CHUNKS = int(SPEECH_TIMEOUT_S / CHUNK_DURATION_S)  # 12
_MIN_CHUNKS     = int(MIN_SPEECH_S / CHUNK_DURATION_S)       # 4

# ── Adaptive VAD — ambient noise calibration ──────────────────────────
# On each mic-on, the engine samples ambient noise for CALIBRATION_S seconds
# and sets the silence threshold to AMBIENT_SNR_RATIO × that ambient RMS.
# This self-tunes to quiet offices, loud labs, or demo environments.
CALIBRATION_S     = 0.8   # seconds to measure ambient noise at session start
AMBIENT_SNR_RATIO = 4.0   # speech must be this many × louder than ambient
_THRESHOLD_MIN    = 0.006 # floor  — never go below (prevents hair-trigger in silence)
_THRESHOLD_MAX    = 0.060 # ceiling — never go above (prevents deaf mode in very loud rooms)
_THRESHOLD_DEFAULT = 0.012 # used if calibration data is unavailable

# ── Voice-send keywords ───────────────────────────────────────────────
SEND_KEYWORDS: frozenset[str] = frozenset({"send", "enter", "go", "submit"})


class SpeechEngine(QObject):
    """
    Manages microphone capture and faster-whisper transcription.

    A lightweight energy-based VAD buffers audio while the user speaks and
    fires Whisper transcription after a silence gap — all in a daemon thread
    so the Qt UI is never blocked.

    All public methods are safe to call from the Qt main thread.
    """

    transcript_partial   = pyqtSignal(str)
    transcript_final     = pyqtSignal(str)
    voice_send_triggered = pyqtSignal()
    listening_started    = pyqtSignal()
    listening_stopped    = pyqtSignal()
    model_status         = pyqtSignal(str)
    error                = pyqtSignal(str)

    def __init__(
        self,
        voice_send_enabled: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._voice_send: bool  = voice_send_enabled
        self._listening: bool   = False
        self._model_ready: bool = False
        self._stop_event        = threading.Event()
        self._model             = None  # faster_whisper.WhisperModel

        threading.Thread(
            target=self._load_model, daemon=True, name="whisper-loader"
        ).start()

    # ── Public API ────────────────────────────────────────────────────

    @property
    def is_listening(self) -> bool:
        return self._listening

    @property
    def voice_send_enabled(self) -> bool:
        return self._voice_send

    @voice_send_enabled.setter
    def voice_send_enabled(self, value: bool) -> None:
        self._voice_send = value

    def start_listening(self) -> None:
        if self._listening:
            return
        if not self._model_ready:
            self.error.emit("Speech model is still loading — please wait.")
            return
        self._listening = True
        self._stop_event.clear()
        threading.Thread(
            target=self._stream_loop, daemon=True, name="whisper-stream"
        ).start()
        self.listening_started.emit()

    def stop_listening(self) -> None:
        if not self._listening:
            return
        self._listening = False
        self._stop_event.set()
        self.listening_stopped.emit()

    def toggle_listening(self) -> None:
        if self._listening:
            self.stop_listening()
        else:
            self.start_listening()

    def shutdown(self) -> None:
        """Call from the main window's closeEvent to release the microphone."""
        self.stop_listening()

    # ── Model loading ─────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            self.error.emit(
                "faster-whisper is not installed.\n"
                "Run:  pip install faster-whisper sounddevice"
            )
            return

        self.model_status.emit("Loading speech detection")
        try:
            from faster_whisper import WhisperModel
            self._model       = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
            self._model_ready = True
            self.model_status.emit("Enter a command")
        except Exception as exc:
            self.error.emit(f"Failed to load speech detection: {exc}")

    # ── Audio stream + VAD + transcription ───────────────────────────

    def _stream_loop(self) -> None:
        try:
            import sounddevice as sd  # noqa: F401
        except ImportError:
            self.error.emit("sounddevice not installed.\nRun:  pip install sounddevice")
            self._listening = False
            self.listening_stopped.emit()
            return

        import sounddevice as sd

        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def _callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            if not self._stop_event.is_set():
                audio_queue.put(indata.copy())

        speech_buffer: List[np.ndarray] = []
        silence_count = 0
        in_speech     = False

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                dtype="int16",
                channels=1,
                callback=_callback,
            ):
                # ── Ambient noise calibration ─────────────────────────
                self.transcript_partial.emit("Calibrating…")
                calibration_chunks = max(1, int(CALIBRATION_S / CHUNK_DURATION_S))
                ambient_rms_samples: List[float] = []
                for _ in range(calibration_chunks):
                    if self._stop_event.is_set():
                        break
                    try:
                        cal_chunk = audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    fc = cal_chunk.astype(np.float32).flatten() / 32768.0
                    ambient_rms_samples.append(float(np.sqrt(np.mean(fc ** 2))))

                if ambient_rms_samples:
                    ambient_rms = float(np.mean(ambient_rms_samples))
                    silence_threshold = max(
                        _THRESHOLD_MIN,
                        min(_THRESHOLD_MAX, ambient_rms * AMBIENT_SNR_RATIO),
                    )
                else:
                    silence_threshold = _THRESHOLD_DEFAULT

                self.transcript_partial.emit("Listening…")

                while not self._stop_event.is_set():
                    try:
                        chunk = audio_queue.get(timeout=0.3)
                    except queue.Empty:
                        continue

                    float_chunk = chunk.astype(np.float32).flatten() / 32768.0
                    rms = float(np.sqrt(np.mean(float_chunk ** 2)))

                    if rms > silence_threshold:
                        if not in_speech:
                            in_speech = True
                            self.transcript_partial.emit("Detecting speech…")
                        silence_count = 0
                        speech_buffer.append(float_chunk)
                    elif in_speech:
                        silence_count += 1
                        speech_buffer.append(float_chunk)
                        if silence_count >= _SILENCE_CHUNKS:
                            if len(speech_buffer) >= _MIN_CHUNKS:
                                self._transcribe(speech_buffer)
                            speech_buffer = []
                            silence_count = 0
                            in_speech     = False

                if in_speech and len(speech_buffer) >= _MIN_CHUNKS:
                    self._transcribe(speech_buffer)

        except Exception as exc:
            self.error.emit(f"Microphone error: {exc}")
            self._listening = False
            self.listening_stopped.emit()

    def _transcribe(self, chunks: List[np.ndarray]) -> None:
        self.transcript_partial.emit("Transcribing…")
        audio = np.concatenate(chunks)
        try:
            segments, _ = self._model.transcribe(
                audio,
                language="en",
                beam_size=1,
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if text:
                self._handle_final(text)
            else:
                self.transcript_partial.emit("Listening…")
        except Exception as exc:
            self.error.emit(f"Transcription error: {exc}")

    def _handle_final(self, text: str) -> None:
        words_lower = {w.lower() for w in text.split()}
        if self._voice_send and words_lower & SEND_KEYWORDS:
            filtered = " ".join(
                w for w in text.split() if w.lower() not in SEND_KEYWORDS
            ).strip()
            if filtered:
                self.transcript_final.emit(filtered)
            self.voice_send_triggered.emit()
        else:
            self.transcript_final.emit(text)
