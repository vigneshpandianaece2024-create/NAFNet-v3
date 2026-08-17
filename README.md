# NAFNet v3


A NAFNet-based grayscale image restoration and 2× super-resolution model.


NAFNet v3 combines a NAFNet restoration body with a PixelShuffle
super-resolution head, a bilinear residual path, and optional TLC
(Tile-based Local Context) inference.


---


## Overview


The goal of this project is to restore noisy low-resolution grayscale
images while simultaneously performing 2× super-resolution.


The main model uses the SIDD-style NAFNet block configuration:


- Encoder blocks: `[2, 2, 4, 8]`
- Middle blocks: `12`
- Decoder blocks: `[2, 2, 2, 2]`
- Width: `32`
- Scale factor: `2×`


The model operates on single-channel grayscale images.


---


## Architecture


```text
                 Noisy LR Input
                       |
                       v
              +------------------+
              |   NAFNet v3 Body |
              |                  |
              | Enc: [2,2,4,8]   |
              | Mid: 12          |
              | Dec: [2,2,2,2]   |
              +--------+---------+
                       |
                       v
              Feature Representation
                       |
                       v
              +------------------+
              | PixelShuffle 2×  |
              |    SR Head       |
              +--------+---------+
                       |
                       v
               Learned SR Residual
                       |
                       +----------+
                                  |
Input LR -----------------> Bilinear 2×
                                  |
                                  v
                         Residual Fusion
                                  |
                                  v
                         Restored 2× Image
NAFBlock

The NAFBlock uses:

LayerNorm2d
Pointwise convolution
Depthwise convolution
SimpleGate
Simplified Channel Attention
Residual scaling

The residual scaling parameters are initialized to zero.

Super-Resolution

The NAFNet restoration body produces feature maps at the input resolution.

A PixelShuffle-based head performs the 2× upsampling.

The final prediction uses a bilinear skip connection:

Final Output =
    Learned SR Residual
    +
    Bilinear 2× Input

This provides a low-frequency reconstruction path while allowing the
network to focus on restoration and high-frequency details.

TLC Inference

During training, channel attention uses global average pooling.

For inference, TLC can replace the global pooling operation with local
window pooling.

The main experiment uses:

Training LR patch: 96 × 96

TLC can be enabled after loading the trained checkpoint.

Dataset

The training pipeline expects the following structure:

combined/
├── GT/
│   ├── image_001.npy
│   ├── image_002.npy
│   └── ...
│
└── NoisyLR/
    ├── image_001.npy
    ├── image_002.npy
    └── ...

The dataset is not included in this repository.

Large datasets and trained checkpoints are intentionally excluded from
Git using .gitignore.

Data Format

The model expects single-channel NumPy arrays.

Supported array layouts include:

H × W
1 × H × W
H × W × 1

The training code handles mixed LR image sizes and groups compatible
crop sizes into batches.

The main experiment uses a 96×96 LR training patch.

Train / Validation Split

Validation is performed at the clean-image level.

This is important when multiple noisy realizations correspond to the same
clean image.

All noisy realizations associated with one clean image remain in the same
split.

This prevents clean-image leakage between training and validation.

A utility is provided at:

utils/split_dataset.py

Example:

python utils/split_dataset.py \
    --root /path/to/combined \
    --output splits \
    --val_frac 0.03 \
    --seed 0
Training

The main experiment was trained for:

100,000 iterations

Configuration:

Parameter	Value
Model width	32
Scale factor	2×
LR patch	96 × 96
Batch size	8
Optimizer	AdamW
Learning rate	3e-4
Weight decay	0
Betas	(0.9, 0.9)
Iterations	100,000
Loss

The training objective is:

L = L1 + 0.1 × Lgradient + 0.1 × LFFT

where:

L1 measures pixel reconstruction error.
Lgradient encourages gradient and edge consistency.
LFFT compares frequency-domain representations.
Training command
python training/train_v3.py \
    --root /path/to/combined \
    --out_dir ckpt_v3 \
    --width 32 \
    --patch 96 \
    --batch 8 \
    --iters 100000

The training script saves:

ckpt_v3/
├── best.pth
└── last.pth

These checkpoint files are ignored by Git.

Inference

Inference is performed using a trained checkpoint.

Example:

python inference/infer_v3.py \
    --ckpt ckpt_v3/best.pth \
    --input /path/to/input \
    --out results
Disable TLC
python inference/infer_v3.py \
    --ckpt ckpt_v3/best.pth \
    --input /path/to/input \
    --out results \
    --no_tlc
Disable test-time ensemble
python inference/infer_v3.py \
    --ckpt ckpt_v3/best.pth \
    --input /path/to/input \
    --out results \
    --no_ensemble
CPU inference
python inference/infer_v3.py \
    --ckpt ckpt_v3/best.pth \
    --input /path/to/input \
    --out results \
    --cpu

The inference script reads architecture metadata stored in the checkpoint
and reconstructs the corresponding NAFNet v3 configuration.

Test-Time Ensemble

The inference pipeline optionally uses an 8-fold geometric ensemble:

4 rotations × 2 flip states = 8 predictions

Each prediction is transformed back to the original orientation and the
predictions are averaged.

This can improve restoration consistency but increases inference cost.

Evaluation

The evaluation script supports PSNR and SSIM.

Example:

python evaluation/evaluate.py \
    --pred results \
    --gt /path/to/ground_truth

The repository also contains a visualization utility:

evaluation/visualize.py

which can generate side-by-side comparisons of:

Input | NAFNet v3 | Ground Truth
NumPy Visualization

NumPy images can be converted to PNG using:

utils/npy_to_png.py

Example:

python utils/npy_to_png.py \
    --input results \
    --output results_png

For a single file:

python utils/npy_to_png.py \
    --input results/image_001.npy \
    --output results_png/image_001.png
Results

The main reported experiment uses a bilinear 2× reconstruction as the
baseline.

Method	PSNR
Bilinear 2× baseline	26 dB
NAFNet v3	29 dB
Improvement	+3 dB
