#!/bin/bash

# NAFNet v3 inference example
#
# Before running:
#   1. Download the trained checkpoint.
#   2. Prepare a directory containing grayscale LR images.
#   3. Update the paths below.

CHECKPOINT="checkpoints/best.pth"
INPUT_DIR="data/NoisyLR"
OUTPUT_DIR="results"

python inference/infer_v3.py \
    --ckpt "$CHECKPOINT" \
    --input "$INPUT_DIR" \
    --out "$OUTPUT_DIR"
