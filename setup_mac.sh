#!/bin/bash
set -e

echo "Setting up auto-video-edit-agent for Apple Silicon..."

if [ "$(uname -m)" != "arm64" ]; then
    echo "This setup requires an Apple Silicon Mac (arm64)"
    exit 1
fi

# Ensure Python 3.11 via Homebrew
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    brew install python@3.11
fi

# Ensure ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    brew install ffmpeg
fi

# Create virtual environment with Python 3.11
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip --quiet
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements-mac.txt

echo ""
echo "Setup complete. Run the app with: ./run_mac.sh"
