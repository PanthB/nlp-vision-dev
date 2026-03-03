# Run order

## 1. SSH into PYNQ and run the pipeline

```bash
ssh xilinx@192.168.2.99
sudo -E /usr/local/share/pynq-venv/bin/python3 /home/xilinx/run_pipeline.py
```

Watch the terminal output — the pipeline will initialize the overlay, start the stream server, and launch the cmd_server UDP listener.

## 2. Start receiver on your Mac

From the capstone-gui repo, with venv activated:

```bash
cd src
python receiver.py
```