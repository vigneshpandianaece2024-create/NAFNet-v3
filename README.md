# NAFNet v3 for Image Restoration and 2× Super-Resolution

A NAFNet-based deep learning model for **grayscale image restoration and 2× super-resolution**.

This project extends the NAFNet architecture with a pixel-shuffle super-resolution head and TLC-style local pooling for inference.

---

## Overview

NAFNet v3 is designed for paired low-resolution noisy image restoration with simultaneous 2× super-resolution.

The model uses a U-shaped NAFNet restoration body followed by a pixel-shuffle super-resolution head.

### Key features

- NAFNet-style U-Net architecture
- Grayscale input and output
- 2× super-resolution
- PixelShuffle reconstruction head
- Bilinear upsampling residual skip
- TLC local pooling for inference
- 8-fold dihedral test-time ensemble
- L1 + gradient + FFT training loss
- Mixed-resolution training support
- Clean-image-level train/validation split

---

## Architecture

The restoration body follows the SIDD-style NAFNet configuration:

```text
Encoder blocks:   [2, 2, 4, 8]
Middle blocks:    12
Decoder blocks:   [2, 2, 2, 2]
Width:            32
