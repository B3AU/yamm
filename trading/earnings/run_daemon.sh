#!/bin/bash
# Run the trading daemon manually (without systemd)
#
# Usage:
#   ./run_daemon.sh           # Normal mode
#   DRY_RUN=true ./run_daemon.sh  # Dry run (no actual orders)

cd /home/beau/yamm

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Create logs directory
mkdir -p logs

# Run daemon
exec python3 -m trading.earnings.daemon
