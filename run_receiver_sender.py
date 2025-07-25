import subprocess
import sys
import os

if __name__ == "__main__":
    receiver_proc = subprocess.Popen([sys.executable, os.path.join("src", "receiver.py")])
    sender_proc = subprocess.Popen([sys.executable, os.path.join("src", "sender.py")])

    try:
        receiver_proc.wait()
    finally:
        sender_proc.terminate()
        sender_proc.wait() 