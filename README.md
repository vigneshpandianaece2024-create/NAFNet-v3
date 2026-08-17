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

The model is based on the NAFNet architecture described in:

Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.

NAFNet v3 modifications

Compared with the reference restoration architecture, this implementation adds:

2× PixelShuffle super-resolution head
Bilinear upsampling skip connection
TLC local pooling during inference
Support for mixed 64×64 and 128×128 LR training images
Model Pipeline
                 Low-Resolution Input
                         │
                         ▼
                 ┌───────────────┐
                 │   NAFNet      │
                 │ Restoration   │
                 │     Body      │
                 └───────┬───────┘
                         │
                         ▼
                  Feature Maps
                         │
                         ▼
                 ┌───────────────┐
                 │ PixelShuffle  │
                 │    SR Head    │
                 └───────┬───────┘
                         │
                         ▼
                 Learned Residual
                         │
                         ├──────────────┐
                         │              │
                         ▼              ▼
                    SR Output     Bilinear 2×
                                      Skip
                         │              │
                         └──────┬───────┘
                                ▼
                         Final Output
Repository Structure
NAFNet-v3/
│
├── models/
│   └── nafnet_v3.py
│
├── training/
│   └── train_v3.py
│
├── inference/
│   └── infer_v3.py
│
├── evaluation/
│
├── configs/
│
├── examples/
│
├── README.md
│
└── requirements.txt
Dataset

The training data consists of paired:

GT/
NoisyLR/

The dataset contains multiple noisy realizations associated with clean images.

The training pipeline supports:

64×64 LR images
128×128 LR images
2× super-resolution targets
multiple noisy realizations per clean image

The validation split is performed at the clean-image level, preventing different noisy realizations of the same clean image from appearing in both training and validation sets.

Training Configuration

The main training configuration used for NAFNet v3 is:

Parameter	Value
Model width	32
Scale factor	2×
Training iterations	100,000
Training patch	96×96 LR
Batch size	8
Optimizer	AdamW
Learning rate	3 × 10⁻⁴
Weight decay	0
Adam betas	(0.9, 0.9)
Scheduler	Cosine Annealing
Minimum learning rate	1 × 10⁻⁷
Gradient clipping	1.0
AMP	Enabled on CUDA
Loss

The training objective is:

L = L1 + 0.1 × Lgradient + 0.1 × LFFT

where:

L1 measures pixel-wise reconstruction error
Lgradient encourages structural and edge consistency
LFFT encourages frequency-domain consistency
Training

The training script is:

training/train_v3.py

Example:

python training/train_v3.py \
    --root /path/to/combined \
    --out_dir checkpoints \
    --width 32 \
    --patch 96 \
    --batch 8 \
    --iters 100000

The training script saves:

checkpoints/
├── best.pth
└── last.pth
Inference

The inference script is:

inference/infer_v3.py

Example:

python inference/infer_v3.py \
    --ckpt checkpoints/best.pth \
    --input /path/to/NoisyLR \
    --out results

With ground-truth images for evaluation:

python inference/infer_v3.py \
    --ckpt checkpoints/best.pth \
    --input /path/to/NoisyLR \
    --gt /path/to/GT \
    --out results
TLC Inference

TLC-style local pooling is enabled by default during inference.

The purpose is to reduce the mismatch between the spatial statistics encountered during training and those encountered when processing larger images.

Disable TLC with:

python inference/infer_v3.py \
    --ckpt checkpoints/best.pth \
    --input /path/to/input \
    --out results \
    --no_tlc
Test-Time Ensemble

Inference uses an optional 8-fold dihedral ensemble:

4 rotations × 2 flip states

This is enabled by default.

To disable it:

python inference/infer_v3.py \
    --ckpt checkpoints/best.pth \
    --input /path/to/input \
    --out results \
    --no_ensemble
Output

For each input image, the inference script can produce:

results/
├── image_000.npy
├── image_000.png
├── image_001.npy
├── image_001.png
└── ...

The .npy output contains the floating-point prediction.

The .png output is clipped to [0, 1] and converted to 8-bit grayscale.

Results

The reported performance for the trained NAFNet v3 model is:

Method	PSNR
Bilinear baseline	26 dB
NAFNet v3	29 dB
Improvement
29 dB - 26 dB = +3 dB

NAFNet v3 therefore provides a reported 3 dB improvement over the bilinear baseline on the evaluated setting.

Note: the 29 dB value is reported as the final NAFNet v3 inference/evaluation result.

Reproducibility

The model configuration is designed to reproduce the NAFNet v3 experiment using:

Width:             32
Scale:             2×
Patch size:        96×96 LR
Iterations:        100,000
Batch size:        8
Learning rate:     3e-4
Optimizer:         AdamW
Loss:              L1 + gradient + FFT

Random seeds are initialized in the training script for Python, NumPy, and PyTorch.

Requirements

The main dependencies are:

Python 3.10+
PyTorch
NumPy
Pillow
LPIPS

A complete requirements.txt is provided separately.

Citation

If you use this implementation or build upon the NAFNet architecture, please cite the original NAFNet paper:

@inproceedings{chen2022simple,
    title={Simple Baselines for Image Restoration},
    author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
    booktitle={European Conference on Computer Vision},
    year={2022}
}
Acknowledgements

This project builds upon the NAFNet image restoration architecture introduced by:

Chen et al., Simple Baselines for Image Restoration, ECCV 2022.

The NAFNet v3 implementation extends the restoration architecture for the combined restoration + 2× super-resolution setting.

License

This repository is intended for research and educational use.

Please review the licenses of the original NAFNet implementation and any datasets used before redistributing derivative code or data.



### 9.4 Don't commit yet


After pasting, make sure the filename says:


```text
README.md
