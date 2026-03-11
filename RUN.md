# Run order

## 0. Deploy stream_server to PYNQ (required after any stream_server change)

The receiver **must** connect to the **TCP** stream_server. If the PYNQ has the wrong binary (e.g. UDP), the receiver will hang spinning.

```bash
./scripts/deploy-stream-server.sh
```

Or manually:
```bash
scp stream_server.c xilinx@192.168.2.99:/tmp/
ssh xilinx@192.168.2.99
cd /tmp && gcc -o stream_server stream_server.c
sudo mv /tmp/stream_server /home/xilinx/stream_server
```

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
