#!/usr/bin/env bash
# Deploy stream_server.c to PYNQ and compile. Run from repo root.
# Usage: ./scripts/deploy-stream-server.sh [user@host]

set -e
HOST="${1:-xilinx@192.168.2.99}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f stream_server.c ]]; then
    echo "Error: stream_server.c not found in $REPO_ROOT"
    exit 1
fi

echo "Deploying stream_server to $HOST ..."
scp stream_server.c "$HOST:/tmp/"

echo "Compiling on PYNQ ..."
ssh "$HOST" \
    'cd /tmp && gcc -o stream_server stream_server.c && sudo mv stream_server /home/xilinx/stream_server && echo "Done. stream_server installed at /home/xilinx/stream_server"'

echo "Deploy complete. Restart the pipeline on the PYNQ."
