# NAFNet v3 - System Architecture

NAFNet v3 is a grayscale image restoration and 2x super-resolution model.
The restoration body follows the SIDD-style NAFNet configuration and adds a
PixelShuffle super-resolution head, a bilinear residual path, and optional
TLC inference.

## 1. Overall Pipeline

```text
Noisy Low-Resolution Image
            |
            v
      NAFNet v3 Body
            |
            v
       Feature Maps
            |
            v
     PixelShuffle 2x
       SR Head
            |
            +-------------------+
            |                   |
            v                   v
   Learned SR Residual    Bilinear 2x Input
            |                   |
            +---------+---------+
                      |
                      v
                Final Output
                      |
                      v
              Restored 2x Image
2. NAFNet v3 Body
 Input
  |
  v
Intro Conv
  |
  v
Encoder 1
2 x NAFBlock
32 channels
  |
  v
Downsample
  |
  v
Encoder 2
2 x NAFBlock
64 channels
  |
  v
Downsample
  |
  v
Encoder 3
4 x NAFBlock
128 channels
  |
  v
Downsample
  |
  v
Encoder 4
8 x NAFBlock
256 channels
  |
  v
Downsample
  |
  v
Middle
12 x NAFBlock
512 channels
  |
  v
Upsample + Skip
  |
  v
Decoder 1
2 x NAFBlock
256 channels
  |
  v
Upsample + Skip
  |
  v
Decoder 2
2 x NAFBlock
128 channels
  |
  v
Upsample + Skip
  |
  v
Decoder 3
2 x NAFBlock
64 channels
  |
  v
Upsample + Skip
  |
  v
Decoder 4
2 x NAFBlock
32 channels
  |
  v
Ending Conv
  |
  v
Feature Output

3. NAFBlock

Each NAFBlock contains LayerNorm2d, pointwise convolution,
depthwise convolution, SimpleGate, simplified channel attention,
and two residual branches.


Input
  |
  v
LayerNorm2d
  |
  v
1x1 Conv
  |
  v
Depthwise 3x3 Conv
  |
  v
SimpleGate
  |
  v
Simplified Channel Attention
  |
  v
1x1 Conv
  |
  v
Beta Residual Scaling
  |
  v
Residual Add
  |
  v
LayerNorm2d
  |
  v
1x1 Conv
  |
  v
SimpleGate
  |
  v
1x1 Conv
  |
  v
Gamma Residual Scaling
  |
  v
Residual Add
  |
  v
Output

4. Super-Resolution Head

The model uses a PixelShuffle-based 2x super-resolution head.

NAFNet Feature Maps
        |
        v
     3x3 Conv
        |
        v
       GELU
        |
        v
     3x3 Conv
        |
        v
   PixelShuffle(2)
        |
        v
     3x3 Conv
        |
        v
       GELU
        |
        v
     3x3 Conv
        |
        v
Learned SR Residual

5. TLC Inference

During training, channel attention uses global average pooling.

During inference, TLC can replace global pooling with local window pooling.

Training:

Feature Map
    |
    v
Global Average Pooling
    |
    v
Channel Attention
Inference with TLC:

Feature Map
    |
    v
Local Window Pooling
    |
    v
Channel Attention

The main training patch size is:
96 x 96 LR pixels
6. Inference Pipeline
Input LR Image
      |
      v
Load Checkpoint
      |
      v
Build NAFNet v3
      |
      v
Enable TLC
      |
      v
Generate Prediction
      |
      v
2x Super-Resolution
      |
      v
Optional 8-Fold Ensemble
      |
      v
Average Predictions
      |
      v
Save Output
7.Evaluation

The evaluation compares the restored image against the ground truth.
Primary metrics:
PSNR
SSIM
Model              : NAFNet v3
Input channels     : 1
Output channels    : 1
Width              : 32
Scale factor       : 2x

Encoder blocks     : [2, 2, 4, 8]
Middle blocks      : 12
Decoder blocks     : [2, 2, 2, 2]

Training patch     : 96 x 96 LR
Training iterations: 100,000
Batch size         : 8

Optimizer          : AdamW
Learning rate      : 3e-4
Weight decay       : 0
Betas              : (0.9, 0.9)

Loss               : L1 + 0.1 Gradient + 0.1 FFT

TLC                : Enabled during inference
TTA                : Optional 8-fold ensemble

Baseline PSNR      : 26 dB
NAFNet v3 PSNR     : 29 dB

Complete System:
                         DATASET
                            |
                            v
                 Clean-Image-Level Split
                            |
                 +----------+----------+
                 |                     |
                 v                     v
              Training             Validation
                 |
                 v
             96x96 LR Crop
                 |
                 v
              NAFNet v3
                 |
          +------+------+
          |             |
          v             v
   Restoration Body   Bilinear 2x
          |             |
          v             |
   PixelShuffle 2x      |
          |             |
          +------+------+
                 |
                 v
            Final Output
                 |
                 v
                 TLC
                 |
                 v
        Optional 8x Ensemble
                 |
                 v
             Evaluation
                 |
          +------+------+
          |      |      |
          v      v      v
        PSNR   SSIM   LPIPS

 Reference

Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.

