#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Set environment variables to prevent Metal GPU conflicts
# Force PyTorch to use CPU fallback instead of Metal GPU acceleration
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Start backend WITHOUT reload to prevent constant restarts
# The --reload was causing issues with MLX Whisper dependencies
uvicorn backend.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  --no-access-log 