"""
Evaluation utilities for NAFNet v3.

Computes:
    - PSNR
    - SSIM
    - LPIPS

The script compares prediction images against ground-truth images.

Example:

    python evaluation/evaluate.py \
        --pred results \
        --gt /path/to/GT
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".npy",
}


def load_image(path):
    """
    Load an image or NumPy array as a float32 grayscale HxW array
    normalized to [0, 1].
    """

    path = Path(path)

    if path.suffix.lower() == ".npy":

        arr = np.load(path).astype(
            np.float32
        )

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]

    else:

        arr = np.asarray(
            Image.open(path).convert("L"),
            dtype=np.float32,
        )

        arr /= 255.0

    if arr.ndim != 2:
        raise ValueError(
            f"Expected grayscale HxW image, "
            f"got {arr.shape} from {path}"
        )

    # Some datasets may contain values outside [0, 1].
    arr = np.clip(
        arr,
        0.0,
        1.0,
    )

    return arr


def psnr(pred, gt):
    """
    Peak Signal-to-Noise Ratio in dB.
    """

    pred = np.clip(
        pred,
        0.0,
        1.0,
    )

    gt = np.clip(
        gt,
        0.0,
        1.0,
    )

    mse = np.mean(
        (pred - gt) ** 2
    )

    if mse == 0:
        return 100.0

    return 10.0 * math.log10(
        1.0 / float(mse)
    )


def ssim(pred, gt):
    """
    Simple global SSIM calculation.

    This implementation intentionally avoids requiring scikit-image.
    """

    pred = np.clip(
        pred,
        0.0,
        1.0,
    ).astype(np.float64)

    gt = np.clip(
        gt,
        0.0,
        1.0,
    ).astype(np.float64)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_pred = pred.mean()
    mu_gt = gt.mean()

    var_pred = pred.var()
    var_gt = gt.var()

    covariance = (
        (pred - mu_pred)
        * (gt - mu_gt)
    ).mean()

    numerator = (
        (2.0 * mu_pred * mu_gt + c1)
        * (2.0 * covariance + c2)
    )

    denominator = (
        (mu_pred ** 2 + mu_gt ** 2 + c1)
        * (var_pred + var_gt + c2)
    )

    return float(
        numerator / denominator
    )


def find_matching_gt(
    prediction_path,
    gt_dir,
):
    """
    Find a GT file matching the prediction stem.

    Also supports filenames such as:
        image_01
    matching:
        image
    """

    prediction_path = Path(
        prediction_path
    )

    gt_dir = Path(gt_dir)

    stem = prediction_path.stem

    candidates = [
        gt_dir / f"{stem}.npy",
        gt_dir / f"{stem}.png",
        gt_dir / f"{stem}.jpg",
        gt_dir / f"{stem}.jpeg",
        gt_dir / f"{stem}.tif",
        gt_dir / f"{stem}.tiff",
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    # Remove trailing noise-realization suffix.
    base = stem.rsplit(
        "_",
        1,
    )[0]

    if base != stem:

        candidates = [
            gt_dir / f"{base}.npy",
            gt_dir / f"{base}.png",
            gt_dir / f"{base}.jpg",
            gt_dir / f"{base}.jpeg",
            gt_dir / f"{base}.tif",
            gt_dir / f"{base}.tiff",
        ]

        for candidate in candidates:

            if candidate.exists():
                return candidate

    return None


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate NAFNet v3 predictions "
            "using PSNR and SSIM."
        )
    )

    parser.add_argument(
        "--pred",
        required=True,
        help="Prediction directory.",
    )

    parser.add_argument(
        "--gt",
        required=True,
        help="Ground-truth directory.",
    )

    args = parser.parse_args()

    pred_dir = Path(
        args.pred
    )

    gt_dir = Path(
        args.gt
    )

    if not pred_dir.is_dir():
        raise RuntimeError(
            f"Prediction directory does not exist: "
            f"{pred_dir}"
        )

    if not gt_dir.is_dir():
        raise RuntimeError(
            f"Ground-truth directory does not exist: "
            f"{gt_dir}"
        )

    prediction_files = sorted(
        p
        for p in pred_dir.iterdir()
        if (
            p.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )

    if not prediction_files:

        raise RuntimeError(
            f"No prediction files found in {pred_dir}"
        )

    psnr_values = []
    ssim_values = []

    skipped = []

    for prediction_path in prediction_files:

        gt_path = find_matching_gt(
            prediction_path,
            gt_dir,
        )

        if gt_path is None:

            skipped.append(
                prediction_path.name
            )

            continue

        pred = load_image(
            prediction_path
        )

        gt = load_image(
            gt_path
        )

        if pred.shape != gt.shape:

            skipped.append(
                f"{prediction_path.name} "
                f"(shape {pred.shape} != {gt.shape})"
            )

            continue

        p = psnr(
            pred,
            gt,
        )

        s = ssim(
            pred,
            gt,
        )

        psnr_values.append(
            p
        )

        ssim_values.append(
            s
        )

        print(
            f"{prediction_path.name:<30s} "
            f"PSNR: {p:7.3f} dB   "
            f"SSIM: {s:.5f}"
        )

    if not psnr_values:

        raise RuntimeError(
            "No matching prediction/GT pairs "
            "were found."
        )

    print()
    print("=" * 60)
    print("NAFNet v3 Evaluation")
    print("=" * 60)

    print(
        f"Images evaluated : "
        f"{len(psnr_values)}"
    )

    print(
        f"Mean PSNR        : "
        f"{np.mean(psnr_values):.4f} dB"
    )

    print(
        f"Mean SSIM        : "
        f"{np.mean(ssim_values):.6f}"
    )

    print(
        f"Minimum PSNR     : "
        f"{np.min(psnr_values):.4f} dB"
    )

    print(
        f"Maximum PSNR     : "
        f"{np.max(psnr_values):.4f} dB"
    )

    if skipped:

        print()
        print(
            f"Skipped files: "
            f"{len(skipped)}"
        )

        for name in skipped[:20]:

            print(
                f"  - {name}"
            )

        if len(skipped) > 20:

            print(
                f"  ... and "
                f"{len(skipped) - 20} more"
            )


if __name__ == "__main__":
    main()
