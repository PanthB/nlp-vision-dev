# Run order

## 1. SSH into PYNQ and start Jupyter

```bash
ssh xilinx@192.168.2.99
cd ~/jupyter_notebooks/Full_Frame_Test
sudo jupyter notebook March1.ipynb --allow-root
```

## 2. Open March1 in browser

On your laptop browser go to the Jupyter URL (e.g. `http://192.168.2.99:9090/` or the URL with token shown in the terminal). Open **March1.ipynb**.

## 3. Run cell 1 in the notebook

Run the first cell. Note the **device address** from the output (you need it for the next step).

## 4. Start stream_server (new SSH terminal)

In a **new** terminal, SSH in again as xilinx, then run (use the hex device address from cell 1 output):

```bash
ssh xilinx@192.168.2.99
sudo /home/xilinx/stream_server <HEX_DEVICE_ADDRESS_FROM_CELL_1>
```

## 5. Start receiver on your Mac

From the capstone-gui repo, with venv activated:

```bash
cd src
python receiver.py
```

## 6. Start cmd_server from the notebook

In March1.ipynb, run the **last cell** (cmd_server UDP listener). You should see something like:

```
[March1] cmd_server running in background on UDP :8888
Send commands from receiver.py or: echo -n 'Turn On Greyscale' | nc -u <PYNQ_IP> 8888
[FilterController] All filters OFF
Initializing High-Accuracy Edge Classifier...
Loaded successfully in 2.51 seconds.
[cmd_server] NLP classifier loaded.
[cmd_server] Mode: NLP → FilterController
[cmd_server] Valid intents: ['blur', 'decrease blue', 'decrease blur', 'decrease green', 'decrease red', 'gaussian', 'grayscale', 'greyscale', 'increase blue', 'increase blur', 'increase green', 'increase red', 'turn off crop', 'turn off greyscale', 'turn on crop', 'turn on greyscale']
[cmd_server] Listening on 0.0.0.0:8888 (UDP)
```

---

## Troubleshooting

### RGB gain makes the display go black and it stays black

**What you see:** After "make it more red" (or any RGB gain change) the video goes black. "Make it less red" or "reset" does not restore the image.

**Likely causes (software vs hardware):**

1. **Unity gain value** – The IP may use **255** for unity, not 128. In that case 128 would be ~half per channel (≈ 0.5³ ≈ 0.125 brightness) and the image would look almost black. Check the IP/RTL docs: is unity gain 128 or 255?
2. **Gains never written at startup** – If "All filters OFF" only clears enable bits and does **not** write 128 to `REG_RED_GAIN` / `REG_GREEN_GAIN` / `REG_BLUE_GAIN`, the hardware may keep 0 or garbage. Then the first RGB enable writes 160,128,128 and the pipeline can end up in a bad state.
3. **No way to fully disable RGB filter** – "Reset all filters" was classified as "Turn Off Greyscale", so the RGB filter stayed enabled. If the pipeline gets stuck when the RGB block is on, you need an explicit "turn off RGB" / "reset gains" to recover.

**Fixes to try in `filter_controller.py` (on the PYNQ):**

- **Initialize gains on startup**  
  In `FilterController.__init__` (or wherever "All filters OFF" is applied), after clearing enable bits, **write 128 to all three gain registers** so the hardware starts from a known state:
  ```python
  self._mmio.write(_REG_RED_GAIN, 128)
  self._mmio.write(_REG_GREEN_GAIN, 128)
  self._mmio.write(_REG_BLUE_GAIN, 128)
  ```
  And set `self._red_gain = self._green_gain = self._blue_gain = 128`.

- **If the IP uses 255 for unity**  
  Change `_GAIN_DEFAULT` (and any initialisation) from 128 to 255, and ensure the step and min/max still make sense (e.g. step 32 with range 0–255).

- **Recovery path**  
  Add a method that disables the RGB filter (clear bit 1 and bits 29/30/31), writes 128 (or 255) to all gain registers, then optionally re-enables the RGB filter. Map it to an intent like "reset all filters" or "turn off rgb" so you can recover without restarting.

- **Quick test**  
  Send "turn off greyscale" then a command that **disables** the RGB filter (if you have one). If the image comes back when the RGB block is off, the bug is in the gain path or the unity value.

**Summary:** It can be either a software issue (wrong unity value, uninitialized gains, no reset) or a hardware quirk (pipeline stuck until RGB is disabled). Start by initializing gains to 128 (or 255) on startup and adding an explicit "reset gains / disable RGB" path; if it still stays black, check the IP datasheet for the correct unity gain and behaviour when the RGB block is enabled.

---

**Summary:** SSH → cd + start Jupyter → open March1 in browser → run cell 1 → new SSH + stream_server with device address → Mac: receiver.py → notebook: run last cell (cmd_server).




LAST CELL:

```
# ── cmd_server: UDP command listener (run after cells 0 & 1: overlay + stream) ──
# Receives natural-language commands from receiver.py and applies filters via FilterController.
# Runs in background so the notebook stays interactive.

import os
import sys
import socket
import threading

# Add user's .local site-packages (onnxruntime is here)
_USER_SITE = "/home/xilinx/.local/lib/python3.10/site-packages"
if os.path.exists(_USER_SITE) and _USER_SITE not in sys.path:
    sys.path.insert(0, _USER_SITE)

# Add pynq root (cmd_server, filter_controller)
_PYNQ_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if _PYNQ_ROOT not in sys.path:
    sys.path.insert(0, _PYNQ_ROOT)


def _run_cmd_server():
    _argv = sys.argv
    sys.argv = ["cmd_server.py"]
    try:
        import cmd_server
        cmd_server.main()
    finally:
        sys.argv = _argv


# Only start if port not already in use (e.g. from a previous run of this cell)
PORT = 8888
try:
    _probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _probe.bind(("", PORT))
    _probe.close()
except OSError:
    print(f"[March1] cmd_server already running on UDP :{PORT} (skip starting again)")
else:
    _thread = threading.Thread(target=_run_cmd_server, daemon=True)
    _thread.start()
    print(f"[March1] cmd_server running in background on UDP :{PORT}")
    print("Send commands from receiver.py or: echo -n 'Turn On Greyscale' | nc -u <PYNQ_IP> 8888")
```