"""
NAFNet v3 inference
===================

NAFNetSR for grayscale image restoration + 2x super-resolution.

Features
--------
TLC:
    Local-window average pooling matched to the training patch size.
    Enabled by default. Disable with --no_tlc.

Ensemble:
    8-fold dihedral test-time augmentation
    (4 rotations x 2 flips).
    Enabled by default. Disable with --no_ensemble.

Scoring:
    PSNR, SSIM and LPIPS when --gt is provided.

Output:
    One PNG and one float32 NPY per input image.

Examples
--------

Blind inference:

    python inference/infer_v3.py \
        --ckpt checkpoints/best.pth \
        --input /path/to/NoisyLR \
        --out results

Scored inference:

    python inference/infer_v3.py \
        --ckpt checkpoints/best.pth \
        --input /path/to/NoisyLR \
        --gt /path/to/GT \
        --out results

Fast inference without TLC or ensemble:

    python inference/infer_v3.py \
        --ckpt checkpoints/best.pth \
        --input /path/to/NoisyLR \
        --out results \
        --no_tlc \
        --no_ensemble
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Make the repository's models/ directory importable
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from nafnet_v3 import (
    NAFNetSR,
    enable_tlc,
    disable_tlc,
)


# ---------------------------------------------------------------------------
# Supported input formats
# ---------------------------------------------------------------------------

EXTS = {
    ".npy",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """
    Load an image as float32 HxW.

    NPY:
        Loaded directly as float32.

    Image files:
        Converted to grayscale and normalized to [0, 1].

    The function also collapses singleton channel dimensions.
    """

    if path.suffix.lower() == ".npy":

        a = np.load(path).astype(
            np.float32
        )

    else:

        a = np.asarray(
            Image.open(path).convert("L"),
            dtype=np.float32,
        ) / 255.0

    # Collapse singleton dimensions.

    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]

    elif a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]

    if a.ndim != 2:
        raise ValueError(
            f"Unexpected array shape "
            f"{a.shape} for {path}"
        )

    return a


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_outputs(
    arr: np.ndarray,
    stem: Path,
    save_npy: bool = True,
):
    """
    Save prediction as:

        <stem>.npy
        <stem>.png

    NPY is stored as float32.

    PNG is clipped to [0, 1] and converted to uint8.
    """

    arr_clipped = np.clip(
        arr,
        0.0,
        1.0,
    )

    if save_npy:

        np.save(
            str(stem) + ".npy",
            arr.astype(np.float32),
        )

    Image.fromarray(
        (
            arr_clipped * 255.0
        )
        .round()
        .astype(np.uint8)
    ).save(
        str(stem) + ".png"
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(
    pred: np.ndarray,
    gt: np.ndarray,
) -> float:

    pred = np.clip(
        pred,
        0,
        1,
    )

    gt = np.clip(
        gt,
        0,
        1,
    )

    mse = float(
        np.mean(
            (pred - gt) ** 2
        )
    )

    if mse == 0:
        return 100.0

    return 10.0 * math.log10(
        1.0 / mse
    )


def ssim_score(
    pred: np.ndarray,
    gt: np.ndarray,
) -> float:
    """
    Simple single-scale SSIM.

    This implementation avoids requiring scikit-image.
    """

    p = np.clip(
        pred,
        0,
        1,
    ).astype(np.float64)

    g = np.clip(
        gt,
        0,
        1,
    ).astype(np.float64)

    C1 = (
        0.01 * 1
    ) ** 2

    C2 = (
        0.03 * 1
    ) ** 2

    mu_p = p.mean()
    mu_g = g.mean()

    sig_p = p.var()
    sig_g = g.var()

    sig_pg = (
        (p - mu_p)
        * (g - mu_g)
    ).mean()

    numerator = (
        (2 * mu_p * mu_g + C1)
        * (2 * sig_pg + C2)
    )

    denominator = (
        (mu_p ** 2 + mu_g ** 2 + C1)
        * (sig_p + sig_g + C2)
    )

    return float(
        numerator / denominator
    )


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

_lpips_fn = None


def lpips_score(
    pred: np.ndarray,
    gt: np.ndarray,
    device: str,
) -> float:
    """
    Compute LPIPS using the AlexNet backbone.

    LPIPS expects 3-channel input in [-1, 1].
    Grayscale images are replicated across three channels.
    """

    global _lpips_fn

    if _lpips_fn is None:

        try:

            import lpips as lpips_lib

        except ImportError:

            import subprocess

            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "lpips",
                    "-q",
                ]
            )

            import lpips as lpips_lib

        _lpips_fn = (
            lpips_lib.LPIPS(
                net="alex",
                verbose=False,
            )
            .to(device)
        )

    def to_tensor(a):

        t = torch.from_numpy(
            np.clip(
                a,
                0,
                1,
            ).astype(
                np.float32
            )
        )

        # [H,W]
        # -> [1,1,H,W]
        # -> [1,3,H,W]
        # -> [-1,1]

        t = (
            t.unsqueeze(0)
            .unsqueeze(0)
            .repeat(
                1,
                3,
                1,
                1,
            )
            * 2.0
            - 1.0
        )

        return t.to(device)

    with torch.no_grad():

        distance = _lpips_fn(
            to_tensor(pred),
            to_tensor(gt),
        )

    return float(
        distance.item()
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(
    ckpt_path: str,
    device: str,
):
    """
    Load NAFNet v3 from a training checkpoint.

    Architecture information is read from the checkpoint when available.
    """

    ck = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    args_from_ckpt = ck.get(
        "args",
        {},
    )

    sf = ck.get(
        "sf",
        args_from_ckpt.get(
            "sf",
            2,
        ),
    )

    width = ck.get(
        "width",
        args_from_ckpt.get(
            "width",
            32,
        ),
    )

    enc_blks = ck.get(
        "enc_blks",
        args_from_ckpt.get(
            "enc_blks",
            [2, 2, 4, 8],
        ),
    )

    middle_blks = ck.get(
        "middle_blks",
        args_from_ckpt.get(
            "middle_blks",
            12,
        ),
    )

    dec_blks = ck.get(
        "dec_blks",
        args_from_ckpt.get(
            "dec_blks",
            [2, 2, 2, 2],
        ),
    )

    train_patch = ck.get(
        "train_patch",
        args_from_ckpt.get(
            "patch",
            96,
        ),
    )

    val_psnr = ck.get(
        "best",
        float("nan"),
    )

    iteration = ck.get(
        "iter",
        "?",
    )

    print(
        f"Checkpoint : {ckpt_path}"
    )

    print(
        f"  iter {iteration}  |  "
        f"val PSNR {val_psnr:.2f} dB"
    )

    print(
        f"  scale factor {sf}x  |  "
        f"width {width}  |  "
        f"train patch {train_patch}"
    )

    print(
        f"  encoder blocks {enc_blks}"
    )

    print(
        f"  middle blocks {middle_blks}"
    )

    print(
        f"  decoder blocks {dec_blks}"
    )

    model = NAFNetSR(
        sf=sf,
        width=width,
        middle_blk_num=middle_blks,
        enc_blk_nums=enc_blks,
        dec_blk_nums=dec_blks,
    ).to(device)

    model.load_state_dict(
        ck["model"]
    )

    model.eval()

    return (
        model,
        sf,
        train_patch,
    )


# ---------------------------------------------------------------------------
# Single-pass inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_single(
    model,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass for one [1,1,H,W] tensor.
    """

    return model(x)


# ---------------------------------------------------------------------------
# 8-fold dihedral ensemble
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_ensemble(
    model,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    8-fold dihedral test-time ensemble:

        4 rotations x 2 flips

    Each prediction is transformed back to the original orientation
    before averaging.
    """

    outputs = []

    for k in range(4):

        for flip in (
            False,
            True,
        ):

            transformed = torch.rot90(
                x,
                k,
                dims=[2, 3],
            )

            if flip:

                transformed = torch.flip(
                    transformed,
                    dims=[3],
                )

            prediction = model(
                transformed
            )

            if flip:

                prediction = torch.flip(
                    prediction,
                    dims=[3],
                )

            prediction = torch.rot90(
                prediction,
                -k,
                dims=[2, 3],
            )

            outputs.append(
                prediction
            )

    return torch.stack(
        outputs
    ).mean(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "NAFNet v3 inference "
            "with TLC and optional "
            "8-fold ensemble."
        )
    )

    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the trained .pth checkpoint.",
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Directory containing input "
            "images/NPY files, or one file."
        ),
    )

    parser.add_argument(
        "--out",
        default="results",
        help="Output directory.",
    )

    parser.add_argument(
        "--gt",
        default=None,
        help=(
            "Optional directory containing "
            "GT files for scoring."
        ),
    )

    parser.add_argument(
        "--no_tlc",
        action="store_true",
        help="Disable TLC.",
    )

    parser.add_argument(
        "--no_ensemble",
        action="store_true",
        help="Disable 8-fold ensemble.",
    )

    parser.add_argument(
        "--no_npy",
        action="store_true",
        help="Save only PNG outputs.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Process only the first N files. "
            "0 means all files."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------

    device = (
        "cpu"
        if args.cpu
        or not torch.cuda.is_available()
        else "cuda"
    )

    save_npy = not args.no_npy

    use_ensemble = not args.no_ensemble

    use_tlc = not args.no_tlc

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------

    model, sf, train_patch = build_model(
        args.ckpt,
        device,
    )

    # -----------------------------------------------------------------------
    # TLC
    # -----------------------------------------------------------------------

    if use_tlc:

        n = enable_tlc(
            model,
            train_patch,
        )

        print(
            f"  TLC: enabled "
            f"({n} blocks, "
            f"window={train_patch}px LR)"
        )

    else:

        disable_tlc(
            model
        )

        print(
            "  TLC: disabled"
        )

    print(
        f"  Ensemble: "
        f"{'8-fold' if use_ensemble else 'disabled'}"
    )

    print(
        f"  Device: {device}"
    )

    print()

    # -----------------------------------------------------------------------
    # Find input files
    # -----------------------------------------------------------------------

    input_path = Path(
        args.input
    )

    if input_path.is_dir():

        files = sorted(
            p
            for p in input_path.rglob("*")
            if (
                p.suffix.lower() in EXTS
                and not p.name.startswith("._")
            )
        )

    else:

        files = [
            input_path
        ]

    if args.limit:

        files = files[
            :args.limit
        ]

    if not files:

        raise RuntimeError(
            f"No supported images found at "
            f"{input_path}"
        )

    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------

    out_dir = Path(
        args.out
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt_dir = (
        Path(args.gt)
        if args.gt
        else None
    )

    # -----------------------------------------------------------------------
    # Metric storage
    # -----------------------------------------------------------------------

    psnr_list = []
    ssim_list = []
    lpips_list = []
    time_list = []

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    for i, file_path in enumerate(
        files
    ):

        lr = load_image(
            file_path
        )

        x = torch.from_numpy(
            lr
        )[None, None].to(
            device
        )

        # ---------------------------------------------------------------
        # Forward pass
        # ---------------------------------------------------------------

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        if use_ensemble:

            y = run_ensemble(
                model,
                x,
            )

        else:

            y = run_single(
                model,
                x,
            )

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.time()
            - start_time
        )

        time_list.append(
            elapsed
        )

        # ---------------------------------------------------------------
        # Convert prediction to numpy
        # ---------------------------------------------------------------

        pred = (
            y[0, 0]
            .clamp(0, 1)
            .cpu()
            .numpy()
        )

        # ---------------------------------------------------------------
        # Save outputs
        # ---------------------------------------------------------------

        save_outputs(
            pred,
            out_dir / file_path.stem,
            save_npy=save_npy,
        )

        line = (
            f"[{i + 1:4d}/{len(files)}] "
            f"{file_path.name:<30s}"
            f"  {lr.shape} -> {pred.shape}"
            f"  {elapsed * 1000:.0f} ms"
        )

        # ---------------------------------------------------------------
        # Optional scoring
        # ---------------------------------------------------------------

        if gt_dir is not None:

            import re

            candidates = [
                p
                for p in gt_dir.iterdir()
                if (
                    p.suffix.lower() in EXTS
                    and p.stem in (
                        file_path.stem,
                        re.sub(
                            r"_\d+$",
                            "",
                            file_path.stem,
                        ),
                    )
                )
            ]

            if candidates:

                gt = load_image(
                    candidates[0]
                )

                p_val = psnr(
                    pred,
                    gt,
                )

                s_val = ssim_score(
                    pred,
                    gt,
                )

                l_val = lpips_score(
                    pred,
                    gt,
                    device,
                )

                psnr_list.append(
                    p_val
                )

                ssim_list.append(
                    s_val
                )

                lpips_list.append(
                    l_val
                )

                line += (
                    f"  PSNR {p_val:.2f}"
                    f"  SSIM {s_val:.4f}"
                    f"  LPIPS {l_val:.4f}"
                )

            else:

                line += (
                    "  [no GT match]"
                )

        print(line)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()

    print(
        f"Saved to: {out_dir}/"
    )

    print(
        f"Speed: "
        f"{np.mean(time_list) * 1000:.1f} ms/image "
        f"(total {sum(time_list):.1f} s "
        f"for {len(files)} images)"
    )

    if use_ensemble:

        print(
            "       "
            "(8-fold ensemble enabled)"
        )

    # -----------------------------------------------------------------------
    # Final metrics
    # -----------------------------------------------------------------------

    if psnr_list:

        print()
        print(
            f"Scores "
            f"({len(psnr_list)} images)"
        )

        print(
            f"  PSNR  : "
            f"{np.mean(psnr_list):.4f} dB "
            f"(min {min(psnr_list):.2f}, "
            f"max {max(psnr_list):.2f})"
        )

        print(
            f"  SSIM  : "
            f"{np.mean(ssim_list):.4f} "
            f"(min {min(ssim_list):.4f}, "
            f"max {max(ssim_list):.4f})"
        )

        print(
            f"  LPIPS : "
            f"{np.mean(lpips_list):.4f} "
            f"(min {min(lpips_list):.4f}, "
            f"max {max(lpips_list):.4f})"
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
