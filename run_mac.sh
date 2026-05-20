#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Run ./setup_mac.sh first"
    exit 1
fi

source .venv/bin/activate

if [ -f "api_keys.env" ]; then
    set -a; source api_keys.env; set +a
fi

export PYTHONPATH="$(pwd)"
echo "Starting (Apple Silicon / MLX mode)..."
python3 app/main.py
