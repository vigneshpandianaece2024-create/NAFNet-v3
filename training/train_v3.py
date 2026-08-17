"""
Train NAFNet v3 on mixed-resolution paired .npy data.

Handles:

1. MIXED IMAGE SIZES
   The combined dataset contains 128x128 and 64x64 LR images.
   With --patch 96, the 64px images can only yield 64px crops.

   A size-aware batch sampler groups images of the same effective crop size
   into each batch. No padding is required and no data is discarded.

2. MULTIPLE NOISE REALISATIONS
   Multiple NoisyLR files can map to the same GT clean image.

3. CLEAN-IMAGE-LEVEL VALIDATION SPLIT
   All noisy realisations belonging to one clean image remain on the same
   side of the train/validation split.

Example:

    python training/train_v3.py \
        --root /path/to/combined \
        --out_dir checkpoints \
        --width 32 \
        --patch 96 \
        --batch 8 \
        --iters 100000
"""

import argparse
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Repository import path
# ---------------------------------------------------------------------------

REPO_ROOT = Path(
    __file__
).resolve().parents[1]

MODELS_DIR = REPO_ROOT / "models"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MODELS_DIR)
    )


# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

from nafnet_v3 import (
    NAFNetSR,
    enable_tlc,
    disable_tlc,
)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def to_hw(a):
    a = np.asarray(a)

    if a.ndim == 2:
        return a

    if a.ndim == 3:

        if a.shape[0] == 1:
            return a[0]

        if a.shape[2] == 1:
            return a[:, :, 0]

    raise ValueError(
        f"Unsupported shape {a.shape}"
    )


def find_pairs(root):
    """
    Returns:
        [(lr_path, gt_path, gt_key)]

    gt_key identifies the clean image so all noisy realizations of the same
    clean image can remain in the same train/validation split.
    """

    gt_d = Path(root) / "GT"
    lr_d = Path(root) / "NoisyLR"

    for d in (gt_d, lr_d):

        if not d.is_dir():

            raise RuntimeError(
                f"Missing folder: {d}"
            )

    gt = {
        p.stem: p
        for p in gt_d.glob("*.npy")
        if not p.name.startswith("._")
    }

    lr_files = sorted(
        p
        for p in lr_d.glob("*.npy")
        if not p.name.startswith("._")
    )

    pairs = []
    unmatched = []

    for p in lr_files:

        if p.stem in gt:

            pairs.append(
                (
                    p,
                    gt[p.stem],
                    p.stem,
                )
            )

            continue

        base = re.sub(
            r"_\d+$",
            "",
            p.stem,
        )

        if base in gt:

            pairs.append(
                (
                    p,
                    gt[base],
                    base,
                )
            )

        else:

            unmatched.append(
                p.name
            )

    if not pairs:

        raise RuntimeError(
            "Nothing matched.\n"
            f"  GT stems      : {sorted(gt)[:5]}\n"
            f"  NoisyLR stems : "
            f"{[p.stem for p in lr_files[:5]]}"
        )

    if unmatched:

        print(
            f"  WARNING {len(unmatched)} unmatched, "
            f"e.g. {unmatched[:3]}"
        )

    n_clean = len(
        {
            k
            for _, _, k in pairs
        }
    )

    print(
        f"  {len(pairs)} pairs from "
        f"{n_clean} clean images "
        f"({len(pairs) / n_clean:.2f} "
        f"noise realisations each)"
    )

    return pairs


def scan_sizes(
    pairs,
    max_probe=4000,
):
    """
    Record each pair's LR size and confirm one consistent scale factor.
    """

    sfs = set()
    seen = {}

    probe = (
        pairs
        if len(pairs) <= max_probe
        else random.sample(
            pairs,
            max_probe,
        )
    )

    for lp, gp, _ in probe:

        l = to_hw(
            np.load(
                lp,
                mmap_mode="r",
            )
        ).shape

        g = to_hw(
            np.load(
                gp,
                mmap_mode="r",
            )
        ).shape

        seen[l] = (
            seen.get(l, 0) + 1
        )

        if (
            g[0] % l[0]
            or g[1] % l[1]
        ):

            raise RuntimeError(
                f"{gp.name}: "
                f"GT {g} not an integer "
                f"multiple of LR {l}"
            )

        sfs.add(
            (
                g[0] // l[0],
                g[1] // l[1],
            )
        )

    if len(sfs) > 1:

        raise RuntimeError(
            f"Mixed scale factors {sfs}. "
            "The model handles ONE scale factor."
        )

    sf = sfs.pop()

    if sf[0] != sf[1]:

        raise RuntimeError(
            f"Non-uniform scale {sf}"
        )

    print(
        f"  LR sizes present: "
        f"{dict(sorted(seen.items()))}"
    )

    print(
        f"  scale factor: "
        f"{sf[0]}x (consistent)"
    )

    sizes = []

    for lp, gp, _ in pairs:

        sizes.append(
            to_hw(
                np.load(
                    lp,
                    mmap_mode="r",
                )
            ).shape
        )

    return sizes, sf[0]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):

    def __init__(
        self,
        pairs,
        sizes,
        patch,
        sf,
        scale,
        train=True,
    ):

        self.pairs = pairs
        self.sizes = sizes
        self.patch = patch
        self.sf = sf
        self.scale = scale
        self.train = train

    def __len__(self):

        return len(
            self.pairs
        )

    def crop_size(self, idx):

        h, w = self.sizes[idx]

        return min(
            self.patch,
            h,
            w,
        )

    def __getitem__(self, idx):

        lp, gp, _ = self.pairs[idx]

        lr = (
            to_hw(
                np.load(lp)
            ).astype(
                np.float32
            )
            / self.scale
        )

        gt = (
            to_hw(
                np.load(gp)
            ).astype(
                np.float32
            )
            / self.scale
        )

        H, W = lr.shape

        p = min(
            self.patch,
            H,
            W,
        )

        if self.train:

            top = random.randint(
                0,
                H - p,
            )

            left = random.randint(
                0,
                W - p,
            )

        else:

            top = (
                H - p
            ) // 2

            left = (
                W - p
            ) // 2

        s = self.sf

        a = torch.from_numpy(
            np.ascontiguousarray(
                lr[
                    top:top + p,
                    left:left + p,
                ]
            )
        )[None]

        b = torch.from_numpy(
            np.ascontiguousarray(
                gt[
                    top * s:(top + p) * s,
                    left * s:(left + p) * s,
                ]
            )
        )[None]

        if self.train:

            if random.random() < 0.5:

                a, b = (
                    torch.flip(
                        a,
                        [2],
                    ),
                    torch.flip(
                        b,
                        [2],
                    ),
                )

            if random.random() < 0.5:

                a, b = (
                    torch.flip(
                        a,
                        [1],
                    ),
                    torch.flip(
                        b,
                        [1],
                    ),
                )

            r = random.randint(
                0,
                3,
            )

            if r:

                a, b = (
                    torch.rot90(
                        a,
                        r,
                        [1, 2],
                    ),
                    torch.rot90(
                        b,
                        r,
                        [1, 2],
                    ),
                )

        return (
            a.contiguous(),
            b.contiguous(),
        )


# ---------------------------------------------------------------------------
# Same-size batch sampler
# ---------------------------------------------------------------------------

class SameSizeBatchSampler(Sampler):

    """
    Groups samples with the same effective crop size into each batch.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        drop_last=True,
    ):

        self.ds = dataset
        self.bs = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = defaultdict(
            list
        )

        for i in range(
            len(dataset)
        ):

            self.buckets[
                dataset.crop_size(i)
            ].append(i)

        counts = {
            k: len(v)
            for k, v in sorted(
                self.buckets.items()
            )
        }

        print(
            f"  crop-size buckets: "
            f"{counts}"
        )

    def __iter__(self):

        batches = []

        for _, idxs in self.buckets.items():

            idxs = idxs[:]

            if self.shuffle:

                random.shuffle(
                    idxs
                )

            for i in range(
                0,
                len(idxs),
                self.bs,
            ):

                b = idxs[
                    i:i + self.bs
                ]

                if (
                    len(b) == self.bs
                    or not self.drop_last
                ):

                    batches.append(
                        b
                    )

        if self.shuffle:

            random.shuffle(
                batches
            )

        return iter(
            batches
        )

    def __len__(self):

        n = 0

        for idxs in self.buckets.values():

            if self.drop_last:

                n += (
                    len(idxs)
                    // self.bs
                )

            else:

                n += math.ceil(
                    len(idxs)
                    / self.bs
                )

        return n


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def gradient_loss(
    p,
    t,
):

    horizontal = F.l1_loss(
        p[..., :, 1:]
        - p[..., :, :-1],

        t[..., :, 1:]
        - t[..., :, :-1],
    )

    vertical = F.l1_loss(
        p[..., 1:, :]
        - p[..., :-1, :],

        t[..., 1:, :]
        - t[..., :-1, :],
    )

    return (
        horizontal
        + vertical
    )


def fft_loss(
    p,
    t,
):

    return F.l1_loss(
        torch.abs(
            torch.fft.rfft2(
                p.float(),
                norm="ortho",
            )
        ),

        torch.abs(
            torch.fft.rfft2(
                t.float(),
                norm="ortho",
            )
        ),
    )


def psnr(
    a,
    b,
):

    mse = F.mse_loss(
        a.clamp(0, 1),
        b.clamp(0, 1),
    ).item()

    if mse == 0:

        return 100.0

    return (
        10
        * math.log10(
            1.0 / mse
        )
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    sf,
):

    model.eval()

    total_psnr = 0.0
    baseline_psnr = 0.0
    n = 0

    for lr, gt in loader:

        lr = lr.to(
            device
        )

        gt = gt.to(
            device
        )

        out = model(
            lr
        )

        baseline = F.interpolate(
            lr,
            scale_factor=sf,
            mode="bilinear",
            align_corners=False,
        )

        for i in range(
            out.size(0)
        ):

            total_psnr += psnr(
                out[i],
                gt[i],
            )

            baseline_psnr += psnr(
                baseline[i],
                gt[i],
            )

            n += 1

    model.train()

    return (
        total_psnr / max(n, 1),
        baseline_psnr / max(n, 1),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        default="ckpt_v3",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--patch",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--lambda_grad",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lambda_fft",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    parser.add_argument(
        "--sf",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "scanning dataset..."
    )

    pairs = find_pairs(
        args.root
    )

    sizes, sf_auto = scan_sizes(
        pairs
    )

    sf = (
        args.sf
        if args.sf > 0
        else sf_auto
    )

    probe = to_hw(
        np.load(
            pairs[0][1]
        )
    )

    scale = (
        255.0
        if probe.max() > 1.5
        else 1.0
    )

    print(
        f"  normalisation divisor: "
        f"{scale}"
    )

    if device == "cuda":

        print(
            f"  GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        torch.backends.cudnn.benchmark = True

    else:

        print(
            "  WARNING: running on CPU, "
            "this will be very slow"
        )

    # -----------------------------------------------------------------------
    # Clean-image-level split
    # -----------------------------------------------------------------------

    keys = sorted(
        {
            k
            for _, _, k in pairs
        }
    )

    random.Random(0).shuffle(
        keys
    )

    n_val_keys = max(
        1,
        int(
            len(keys)
            * args.val_frac
        ),
    )

    val_keys = set(
        keys[:n_val_keys]
    )

    tr_idx = [
        i
        for i, (_, _, k)
        in enumerate(pairs)
        if k not in val_keys
    ]

    va_idx = [
        i
        for i, (_, _, k)
        in enumerate(pairs)
        if k in val_keys
    ]

    tr_pairs = [
        pairs[i]
        for i in tr_idx
    ]

    tr_sizes = [
        sizes[i]
        for i in tr_idx
    ]

    va_pairs = [
        pairs[i]
        for i in va_idx
    ]

    va_sizes = [
        sizes[i]
        for i in va_idx
    ]

    print(
        f"  train {len(tr_pairs)} pairs | "
        f"val {len(va_pairs)} pairs "
        f"from {len(val_keys)} "
        f"held-out clean images"
    )

    # -----------------------------------------------------------------------
    # Data loaders
    # -----------------------------------------------------------------------

    train_ds = PairDataset(
        tr_pairs,
        tr_sizes,
        args.patch,
        sf,
        scale,
        True,
    )

    val_ds = PairDataset(
        va_pairs,
        va_sizes,
        args.patch,
        sf,
        scale,
        False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=SameSizeBatchSampler(
            train_ds,
            args.batch,
        ),
        num_workers=args.workers,
        pin_memory=(
            device == "cuda"
        ),
        persistent_workers=(
            args.workers > 0
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_sampler=SameSizeBatchSampler(
            val_ds,
            max(
                1,
                args.batch // 2,
            ),
            shuffle=False,
            drop_last=False,
        ),
        num_workers=args.workers,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = NAFNetSR(
        sf=sf,
        width=args.width,
    ).to(device)

    disable_tlc(
        model
    )

    print(
        f"\nparams "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
        f"(width {args.width}, "
        f"[2,2,4,8]/12/[2,2,2,2])"
    )

    print(
        f"loss = L1 + "
        f"{args.lambda_grad}*grad + "
        f"{args.lambda_fft}*fft"
    )

    print(
        f"AdamW lr {args.lr}, "
        f"wd {args.weight_decay}, "
        f"betas (0.9, 0.9)\n"
    )

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.9),
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.iters,
            eta_min=1e-7,
        )
    )

    use_amp = (
        device == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # -----------------------------------------------------------------------
    # Checkpoints
    # -----------------------------------------------------------------------

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    iteration = 0
    best = -1.0

    if args.resume:

        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["opt"]
        )

        scheduler.load_state_dict(
            checkpoint["sched"]
        )

        iteration = checkpoint[
            "iter"
        ]

        best = checkpoint.get(
            "best",
            -1.0,
        )

        print(
            f"resumed @ {iteration}, "
            f"best {best:.2f} dB"
        )

    # -----------------------------------------------------------------------
    # Initial validation
    # -----------------------------------------------------------------------

    val_psnr, baseline_psnr = evaluate(
        model,
        val_loader,
        device,
        sf,
    )

    print(
        f"before training: "
        f"{val_psnr:.2f} dB | "
        f"bilinear baseline "
        f"{baseline_psnr:.2f} dB\n"
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    start_time = time.time()

    running_l1 = 0.0
    running_grad = 0.0
    running_fft = 0.0

    while iteration < args.iters:

        for lr_img, gt_img in train_loader:

            if iteration >= args.iters:
                break

            lr_img = lr_img.to(
                device,
                non_blocking=True,
            )

            gt_img = gt_img.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                "cuda",
                enabled=use_amp,
            ):

                output = model(
                    lr_img
                )

                loss_l1 = F.l1_loss(
                    output,
                    gt_img,
                )

                loss_grad = gradient_loss(
                    output,
                    gt_img,
                )

                loss_fft = fft_loss(
                    output,
                    gt_img,
                )

                loss = (
                    loss_l1
                    + args.lambda_grad
                    * loss_grad
                    + args.lambda_fft
                    * loss_fft
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            scheduler.step()

            running_l1 += (
                loss_l1.item()
            )

            running_grad += (
                loss_grad.detach().item()
            )

            running_fft += (
                loss_fft.detach().item()
            )

            iteration += 1

            # ---------------------------------------------------------------
            # Training log
            # ---------------------------------------------------------------

            if iteration % 100 == 0:

                elapsed = (
                    time.time()
                    - start_time
                )

                print(
                    f"iter {iteration:7d}  "
                    f"L1 {running_l1 / 100:.4f}  "
                    f"grad {running_grad / 100:.4f}  "
                    f"fft {running_fft / 100:.4f}  "
                    f"lr {scheduler.get_last_lr()[0]:.2e}  "
                    f"{elapsed / 100:.3f}s/it"
                )

                running_l1 = 0.0
                running_grad = 0.0
                running_fft = 0.0

                start_time = time.time()

            # ---------------------------------------------------------------
            # Validation and checkpoint
            # ---------------------------------------------------------------

            if iteration % args.eval_every == 0:

                val_psnr, baseline_psnr = evaluate(
                    model,
                    val_loader,
                    device,
                    sf,
                )

                checkpoint = {
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "sched": scheduler.state_dict(),
                    "iter": iteration,
                    "best": best,
                    "sf": sf,
                    "scale": scale,
                    "train_patch": args.patch,
                    "width": args.width,
                    "enc_blks": [
                        2,
                        2,
                        4,
                        8,
                    ],
                    "middle_blks": 12,
                    "dec_blks": [
                        2,
                        2,
                        2,
                        2,
                    ],
                    "args": vars(args),
                }

                flag = ""

                if val_psnr > best:

                    best = val_psnr

                    checkpoint[
                        "best"
                    ] = best

                    torch.save(
                        checkpoint,
                        os.path.join(
                            args.out_dir,
                            "best.pth",
                        ),
                    )

                    flag = "  *saved*"

                torch.save(
                    checkpoint,
                    os.path.join(
                        args.out_dir,
                        "last.pth",
                    ),
                )

                print(
                    f"  >> val "
                    f"{val_psnr:.2f} dB  "
                    f"(baseline "
                    f"{baseline_psnr:.2f}, "
                    f"best "
                    f"{best:.2f})"
                    f"{flag}"
                )

                start_time = time.time()

    print(
        f"\ndone. "
        f"best val PSNR "
        f"{best:.2f} dB"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
