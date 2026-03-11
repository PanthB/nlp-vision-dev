#!/usr/bin/env python3
"""
FilterController — AXI register abstraction for the NLP-VisionRT filter pipeline.
VERSION: 9.0 (Fixed Black Screen + Full Dispatch Sync)
"""

from __future__ import annotations
import time

# ──────────────────────────────────────────────────────────────────────────────
# Register Offsets
# ──────────────────────────────────────────────────────────────────────────────
_REG_ENABLE       = 0x00
_REG_ROI_MIN      = 0x04
_REG_HSV_THRESH0  = 0x10
_REG_HSV_THRESH1  = 0x14
_REG_OVERLAY_CLR  = 0x18
_REG_RED_GAIN     = 0x1C
_REG_GREEN_GAIN   = 0x20
_REG_BLUE_GAIN    = 0x24

# REG_ENABLE bit positions
_BIT_ROI      = 0
_BIT_RGB      = 1
_BIT_GREY     = 2
_BIT_BLUR     = 3
_BIT_THRESH   = 4
_BIT_OVERLAY  = 5

# RGB MODES
MODE_GAIN_ONLY  = 0  
MODE_INVERT     = 4  
MODE_DISCO      = 5  # Using RB Swap for Disco
_GAIN_STEP    = 32
_GAIN_DEFAULT = 128  # 128 is NEUTRAL. 0 is BLACK.

# Blur alpha: bits [15:8] of REG_ENABLE. 0=no blur, 255=full blur. NEVER use 0.
_BLUR_STEP     = 32
_BLUR_MIN      = 1
_BLUR_DEFAULT  = 128

# ──────────────────────────────────────────────────────────────────────────────
# FilterController Class
# ──────────────────────────────────────────────────────────────────────────────
class FilterController:
    FILTER_BASE = 0x60000000
    FILTER_SIZE = 0x10000

    _HSV_PRESETS = {
        "red":    (170, 10, 150, 255,  80, 255),
        "green":  (35,  85,  90, 255,  60, 255), 
        "blue":   (100, 130, 90, 255,  60, 255),
        "yellow": (30,  60,  90, 255,  90, 255),
        "white":  (0, 255,   0,  60, 200, 255),
        "black":  (0, 255,   0, 255,   0,  50),
        "gray":   (0, 255,   0,  50,  60, 200),
    }

    def __init__(self, dry_run: bool = False) -> None:
        print(f"[FilterController] __init__ (DryRun={dry_run})")
        try:
            from pynq import MMIO
            self._mmio = MMIO(self.FILTER_BASE, self.FILTER_SIZE)
            print(f"[FilterController] Hardware MMIO Connected.")
        except Exception as e:
            print(f"[FilterController] MMIO FAILED: {e}. Falling back to Mock.")
            self._mmio = _MockMMIO()

        self._history = []
        self._last_method = None
        self._last_arg = None
        self._in_repeat = False

        # --- Default State Initialization ---
        self._roi_on = self._rgb_on = self._grey_on = False
        self._blur_on = self._thresh_on = self._overlay_on = False
        self._rgb_mode = MODE_GAIN_ONLY
        
        # CRITICAL: These must be 128 for a visible picture
        self._red_gain = _GAIN_DEFAULT
        self._green_gain = _GAIN_DEFAULT
        self._blue_gain = _GAIN_DEFAULT
        
        self._roi_x_min, self._roi_y_min = 160, 120
        self._h_min, self._h_max = 170, 10
        self._s_min, self._s_max = 150, 255
        self._v_min, self._v_max = 80, 255
        self._overlay_color = 0x07E0
        self._blur_alpha = _BLUR_DEFAULT  # 1-255, never 0

        # Push initial "clean" state to hardware
        self._sync_all(save_history=False)

    def _sync_all(self, save_history=True) -> None:
        en = (1 << _BIT_ROI if self._roi_on else 0) | \
             (1 << _BIT_RGB if self._rgb_on else 0) | \
             (1 << _BIT_GREY if self._grey_on else 0) | \
             (1 << _BIT_BLUR if self._blur_on else 0) | \
             (1 << _BIT_THRESH if self._thresh_on else 0) | \
             (1 << _BIT_OVERLAY if self._overlay_on else 0) | \
             ((self._blur_alpha & 0xFF) << 8) | \
             ((self._rgb_mode & 0x7) << 29)
        
        # Explicitly writing gains. If these were 0, screen = black.
        self._mmio.write(_REG_ENABLE, en)
        self._mmio.write(_REG_RED_GAIN, int(self._red_gain) & 0xFF)
        self._mmio.write(_REG_GREEN_GAIN, int(self._green_gain) & 0xFF)
        self._mmio.write(_REG_BLUE_GAIN, int(self._blue_gain) & 0xFF)
        
        t0 = ((self._s_max & 0xFF) << 24) | ((self._s_min & 0xFF) << 16) | \
             ((self._h_max & 0xFF) << 8) | (self._h_min & 0xFF)
        self._mmio.write(_REG_HSV_THRESH0, t0)
        self._mmio.write(_REG_HSV_THRESH1, ((self._v_max & 0xFF) << 8) | (self._v_min & 0xFF))
        self._mmio.write(_REG_ROI_MIN, ((int(self._roi_y_min) & 0x3FF) << 10) | (int(self._roi_x_min) & 0x3FF))
        self._mmio.write(_REG_OVERLAY_CLR, int(self._overlay_color) & 0xFFFF)

    def _pre_cmd(self):
        state = {
            'roi': self._roi_on, 'rgb': self._rgb_on, 'grey': self._grey_on,
            'blur': self._blur_on, 'blur_alpha': self._blur_alpha,
            'thresh': self._thresh_on, 'overlay': self._overlay_on,
            'mode': self._rgb_mode, 'r': self._red_gain, 'g': self._green_gain, 'b': self._blue_gain,
            'hmin': self._h_min, 'hmax': self._h_max, 'smin': self._s_min, 'smax': self._s_max,
            'vmin': self._v_min, 'vmax': self._v_max
        }
        self._history.append(state)
        if len(self._history) > 20: self._history.pop(0)

    # ─── API Methods ──────────────────────────────────────────────────────────
    def undo(self):
        print("[FilterController] UNDO: Reverting state.")
        if not self._history: return
        s = self._history.pop()
        self._roi_on, self._rgb_on, self._grey_on = s['roi'], s['rgb'], s['grey']
        self._blur_on, self._thresh_on, self._overlay_on = s['blur'], s['thresh'], s['overlay']
        self._blur_alpha = s.get('blur_alpha', _BLUR_DEFAULT)
        self._rgb_mode, self._red_gain, self._green_gain, self._blue_gain = s['mode'], s['r'], s['g'], s['b']
        self._h_min, self._h_max, self._s_min, self._s_max = s['hmin'], s['hmax'], s['smin'], s['smax']
        self._v_min, self._v_max = s['vmin'], s['vmax']
        self._sync_all(save_history=False)

    def repeat_last(self):
        if not self._last_method: return
        print(f"[FilterController] REPEAT: {self._last_method}")
        self._in_repeat = True
        try:
            m = getattr(self, self._last_method)
            m(self._last_arg) if self._last_arg else m()
        finally: self._in_repeat = False

    def all_off(self): 
        print("[FilterController] RESET: Restoring clear video.")
        self._pre_cmd()
        self._roi_on = self._rgb_on = self._grey_on = self._blur_on = self._thresh_on = self._overlay_on = False
        self._rgb_mode = MODE_GAIN_ONLY
        self._red_gain = self._green_gain = self._blue_gain = _GAIN_DEFAULT
        self._sync_all()

    def enable_greyscale(self):  print("[FilterController] GREYSCALE: ON"); self._pre_cmd(); self._grey_on = True; self._sync_all()
    def disable_greyscale(self): print("[FilterController] GREYSCALE: OFF"); self._pre_cmd(); self._grey_on = False; self._sync_all()
    def enable_blur(self):       print("[FilterController] BLUR: ON"); self._pre_cmd(); self._blur_on = True; self._sync_all()
    def disable_blur(self):      print("[FilterController] BLUR: OFF"); self._pre_cmd(); self._blur_on = False; self._sync_all()
    def increase_blur(self):     self._pre_cmd(); self._blur_alpha = min(255, self._blur_alpha + _BLUR_STEP); self._blur_on = True; print(f"[FilterController] BLUR: + (α={self._blur_alpha})"); self._sync_all()
    def decrease_blur(self):    self._pre_cmd(); self._blur_alpha = max(_BLUR_MIN, self._blur_alpha - _BLUR_STEP); self._blur_on = True; print(f"[FilterController] BLUR: - (α={self._blur_alpha})"); self._sync_all()
    def enable_roi(self):        print("[FilterController] ROI: ON"); self._pre_cmd(); self._roi_on = True; self._sync_all()
    def disable_roi(self):       print("[FilterController] ROI: OFF"); self._pre_cmd(); self._roi_on = False; self._sync_all()
    
    def increase_red(self):   self._pre_cmd(); self._red_gain = min(255, self._red_gain + _GAIN_STEP); self._rgb_on = True; self._sync_all()
    def decrease_red(self):   self._pre_cmd(); self._red_gain = max(0, self._red_gain - _GAIN_STEP); self._rgb_on = True; self._sync_all()
    def increase_green(self): self._pre_cmd(); self._green_gain = min(255, self._green_gain + _GAIN_STEP); self._rgb_on = True; self._sync_all()
    def decrease_green(self): self._pre_cmd(); self._green_gain = max(0, self._green_gain - _GAIN_STEP); self._rgb_on = True; self._sync_all()
    def increase_blue(self):  self._pre_cmd(); self._blue_gain = min(255, self._blue_gain + _GAIN_STEP); self._rgb_on = True; self._sync_all()
    def decrease_blue(self):  self._pre_cmd(); self._blue_gain = max(0, self._blue_gain - _GAIN_STEP); self._rgb_on = True; self._sync_all()

    def enable_disco(self):  print("[FilterController] DISCO: ON"); self._pre_cmd(); self._rgb_mode, self._rgb_on = MODE_DISCO, True; self._sync_all()
    def disable_disco(self): print("[FilterController] DISCO: OFF"); self._pre_cmd(); self._rgb_mode = MODE_GAIN_ONLY; self._sync_all()
    def set_mode_invert(self):  print("[FilterController] INVERT: ON"); self._pre_cmd(); self._rgb_mode, self._rgb_on = MODE_INVERT, True; self._sync_all()
    def disable_invert(self):   print("[FilterController] INVERT: OFF"); self._pre_cmd(); self._rgb_mode = MODE_GAIN_ONLY; self._sync_all()

    def detect_color(self, color_name: str):
        print(f"[FilterController] TRACKING: {color_name.upper()}")
        preset = self._HSV_PRESETS.get(color_name.lower())
        if preset:
            self._pre_cmd()
            self._h_min, self._h_max, self._s_min, self._s_max, self._v_min, self._v_max = preset
            self._thresh_on = self._overlay_on = True
            self._sync_all()

    def undetect_color(self, color_name: str):
        print(f"[FilterController] STOP TRACKING: {color_name.upper()}")
        self._pre_cmd(); self._thresh_on = self._overlay_on = False; self._sync_all()

    # ─── NLP Dispatch ─────────────────────────────────────────────────────────
    _DISPATCH_TABLE = {
        "decrease blue": ("decrease_blue", None),
        "decrease blur": ("decrease_blur", None),
        "decrease green": ("decrease_green", None),
        "decrease red": ("decrease_red", None),
        "detect blue": ("detect_color", "blue"),
        "detect green": ("detect_color", "green"),
        "detect red": ("detect_color", "red"),
        "disable disco": ("disable_disco", None),
        "enable disco": ("enable_disco", None),
        "increase blue": ("increase_blue", None),
        "increase blur": ("increase_blur", None),
        "increase green": ("increase_green", None),
        "increase red": ("increase_red", None),
        "repeat": ("repeat_last", None),
        "reset": ("all_off", None),
        "turn off crop": ("disable_roi", None),
        "turn off greyscale": ("disable_greyscale", None),
        "turn off invert": ("disable_invert", None),
        "turn on crop": ("enable_roi", None),
        "turn on greyscale": ("enable_greyscale", None),
        "turn on invert": ("set_mode_invert", None),
        "undetect blue": ("undetect_color", "blue"),
        "undetect green": ("undetect_color", "green"),
        "undetect red": ("undetect_color", "red"),
        "undo": ("undo", None)
    }

    def dispatch(self, intent: str) -> bool:
        key = intent.strip().lower()
        map = self._DISPATCH_TABLE.get(key)
        if not map: return False
        
        m_name, arg = map
        try:
            m = getattr(self, m_name)
            m(arg) if arg is not None else m()
            if m_name not in ["undo", "repeat_last"] and not self._in_repeat:
                self._last_method, self._last_arg = m_name, arg
            return True
        except Exception as e:
            print(f"[FilterController ERR] {e}")
            return False

class _MockMMIO:
    def write(self, addr, val): pass
    def read(self, addr): return 0