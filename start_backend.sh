#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Start backend WITHOUT reload to prevent constant restarts
# The --reload was causing issues with MLX Whisper dependencies
uvicorn backend.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info 