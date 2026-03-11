#!/usr/bin/env bash
# Sync this workspace to the PYNQ board. Run from repo root.
# Usage: ./scripts/sync-to-pynq.sh [user@host]
# Example: ./scripts/sync-to-pynq.sh xilinx@192.168.2.99

set -e
HOST="${1:-xilinx@192.168.2.99}"
REMOTE_DIR="${2:-/home/xilinx/capstone-gui}"  # adjust if your board uses another path

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Syncing to $HOST:$REMOTE_DIR ..."
rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.git' --exclude 'src/logs/' \
  "$REPO_ROOT/" "$HOST:$REMOTE_DIR/"

echo "Done. SSH and run with: ssh $HOST 'cd $REMOTE_DIR && source venv/bin/activate && python src/receiver.py'"
