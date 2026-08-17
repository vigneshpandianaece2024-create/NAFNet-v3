"""
Train NAFNet v3 on mixed-resolution paired .npy data.

Handles three things the earlier scripts did not:

1. MIXED IMAGE SIZES. The combined dataset has 128x128 and 64x64 LR images.
   With --patch 96 the 64px images can only yield 64px crops, and stacking
   96px and 64px tensors into one batch raises a RuntimeError. A size-aware
   batch sampler groups images of the same size into each batch, so every
   batch is internally uniform while batches may differ from one another.
   Nothing is padded and no data is discarded.

2. MULTIPLE NOISE REALISATIONS. NoisyLR/000000_00.npy and 000000_01.npy both
   map to GT/000000.npy. Each becomes its own pair.

3. VALIDATION SPLIT BY CLEAN IMAGE, not by pair. If the two noise versions
   of one image landed on opposite sides of the split, the validation score
   would be inflated - the model would have trained on that exact content.

Config follows options/train/SIDD/NAFNet-width32.yml:
[2,2,4,8] encoder, 12 middle blocks, [2,2,2,2] decoder,
AdamW betas (0.9, 0.9), weight_decay 0.

Usage:

    python train_v3.py --root /kaggle/working/combined \
        --out_dir /kaggle/working/ckpt_v3 \
        --width 32 --patch 96 --batch 8 --iters 100000
"""

import argparse
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from nafnet_v3 import NAFNetSR, enable_tlc, disable_tlc


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

    raise ValueError(f"Unsupported shape {a.shape}")


def find_pairs(root):
    """
    Returns [(lr_path, gt_path, gt_key)].

    gt_key identifies the clean image, so pairs sharing a GT can be kept on
    the same side of the train/val split.
    """

    gt_d = Path(root) / "GT"
    lr_d = Path(root) / "NoisyLR"

    for d in (gt_d, lr_d):
        if not d.is_dir():
            raise RuntimeError(f"Missing folder: {d}")

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

        base = re.sub(r"_\d+$", "", p.stem)

        if base in gt:
            pairs.append(
                (
                    p,
                    gt[base],
                    base,
                )
            )
        else:
            unmatched.append(p.name)

    if not pairs:
        raise RuntimeError(
            "Nothing matched.\n"
            f"  GT stems      : {sorted(gt)[:5]}\n"
            f"  NoisyLR stems : {[p.stem for p in lr_files[:5]]}"
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
        f"  {len(pairs)} pairs from {n_clean} clean images "
        f"({len(pairs) / n_clean:.2f} noise realisations each)"
    )

    return pairs


def scan_sizes(pairs, max_probe=4000):
    """
    Record each pair's LR size, and confirm one consistent scale factor.
    """

    sizes = []
    sfs = set()

    probe = (
        pairs
        if len(pairs) <= max_probe
        else random.sample(pairs, max_probe)
    )

    seen = {}

    for lp, gp, _ in probe:

        l = to_hw(
            np.load(
                lp,
                mmap_mode="r"
            )
        ).shape

        g = to_hw(
            np.load(
                gp,
                mmap_mode="r"
            )
        ).shape

        seen[l] = seen.get(l, 0) + 1

        if g[0] % l[0] or g[1] % l[1]:
            raise RuntimeError(
                f"{gp.name}: GT {g} not an integer multiple of LR {l}"
            )

        sfs.add(
            (
                g[0] // l[0],
                g[1] // l[1]
            )
        )

    if len(sfs) > 1:
        raise RuntimeError(
            f"Mixed scale factors {sfs}. "
            "The model handles ONE scale factor; "
            "separate the data or retrain per scale."
        )

    sf = sfs.pop()

    if sf[0] != sf[1]:
        raise RuntimeError(
            f"Non-uniform scale {sf}"
        )

    print(
        f"  LR sizes present: {dict(sorted(seen.items()))}"
    )

    print(
        f"  scale factor: {sf[0]}x (consistent)"
    )

    for lp, gp, k in pairs:
        sizes.append(
            to_hw(
                np.load(
                    lp,
                    mmap_mode="r"
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
        return len(self.pairs)

    def crop_size(self, idx):
        h, w = self.sizes[idx]
        return min(
            self.patch,
            h,
            w
        )

    def __getitem__(self, idx):

        lp, gp, _ = self.pairs[idx]

        lr = (
            to_hw(
                np.load(lp)
            ).astype(np.float32)
            / self.scale
        )

        gt = (
            to_hw(
                np.load(gp)
            ).astype(np.float32)
            / self.scale
        )

        H, W = lr.shape

        p = min(
            self.patch,
            H,
            W
        )

        if self.train:
            top = random.randint(
                0,
                H - p
            )

            left = random.randint(
                0,
                W - p
            )
        else:
            top = (H - p) // 2
            left = (W - p) // 2

        s = self.sf

        a = torch.from_numpy(
            np.ascontiguousarray(
                lr[
                    top:top + p,
                    left:left + p
                ]
            )
        )[None]

        b = torch.from_numpy(
            np.ascontiguousarray(
                gt[
                    top * s:(top + p) * s,
                    left * s:(left + p) * s
                ]
            )
        )[None]

        if self.train:

            if random.random() < 0.5:
                a, b = (
                    torch.flip(a, [2]),
                    torch.flip(b, [2])
                )

            if random.random() < 0.5:
                a, b = (
                    torch.flip(a, [1]),
                    torch.flip(b, [1])
                )

            r = random.randint(
                0,
                3
            )

            if r:
                a, b = (
                    torch.rot90(
                        a,
                        r,
                        [1, 2]
                    ),
                    torch.rot90(
                        b,
                        r,
                        [1, 2]
                    )
                )

        return (
            a.contiguous(),
            b.contiguous()
        )


# ---------------------------------------------------------------------------
# Same-size batch sampler
# ---------------------------------------------------------------------------

class SameSizeBatchSampler(Sampler):

    """
    Yields batches whose members all produce the same crop size.

    Indices are bucketed by effective crop size, each bucket is shuffled and
    chunked into batches, then the batches themselves are shuffled so the
    model does not see all 64px batches before all 96px ones.
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

        self.buckets = defaultdict(list)

        for i in range(len(dataset)):
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
            f"  crop-size buckets: {counts}"
        )

    def __iter__(self):

        batches = []

        for _, idxs in self.buckets.items():

            idxs = idxs[:]

            if self.shuffle:
                random.shuffle(idxs)

            for i in range(
                0,
                len(idxs),
                self.bs
            ):

                b = idxs[
                    i:i + self.bs
                ]

                if (
                    len(b) == self.bs
                    or not self.drop_last
                ):
                    batches.append(b)

        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self):

        n = 0

        for idxs in self.buckets.values():

            if self.drop_last:
                n += len(idxs) // self.bs
            else:
                n += math.ceil(
                    len(idxs) / self.bs
                )

        return n


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def gradient_loss(p, t):

    horizontal = F.l1_loss(
        p[..., :, 1:] - p[..., :, :-1],
        t[..., :, 1:] - t[..., :, :-1]
    )

    vertical = F.l1_loss(
        p[..., 1:, :] - p[..., :-1, :],
        t[..., 1:, :] - t[..., :-1, :]
    )

    return horizontal + vertical


def fft_loss(p, t):

    return F.l1_loss(
        torch.abs(
            torch.fft.rfft2(
                p.float(),
                norm="ortho"
            )
        ),
        torch.abs(
            torch.fft.rfft2(
                t.float(),
                norm="ortho"
            )
        )
    )


def psnr(a, b):

    mse = F.mse_loss(
        a.clamp(0, 1),
        b.clamp(0, 1)
    ).item()

    if mse == 0:
        return 100.0

    return 10 * math.log10(
        1.0 / mse
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

    tot = 0
    base = 0
    n = 0

    for lr, gt in loader:

        lr = lr.to(device)
        gt = gt.to(device)

        out = model(lr)

        bas = F.interpolate(
            lr,
            scale_factor=sf,
            mode="bilinear",
            align_corners=False
        )

        for i in range(
            out.size(0)
        ):

            tot += psnr(
                out[i],
                gt[i]
            )

            base += psnr(
                bas[i],
                gt[i]
            )

            n += 1

    model.train()

    return (
        tot / max(n, 1),
        base / max(n, 1)
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True
    )

    ap.add_argument(
        "--out_dir",
        default="ckpt_v3"
    )

    ap.add_argument(
        "--width",
        type=int,
        default=32
    )

    ap.add_argument(
        "--patch",
        type=int,
        default=96
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=8
    )

    ap.add_argument(
        "--iters",
        type=int,
        default=100000
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    ap.add_argument(
        "--weight_decay",
        type=float,
        default=0.0
    )

    ap.add_argument(
        "--lambda_grad",
        type=float,
        default=0.1
    )

    ap.add_argument(
        "--lambda_fft",
        type=float,
        default=0.1
    )

    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.03
    )

    ap.add_argument(
        "--eval_every",
        type=int,
        default=1000
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=2
    )

    ap.add_argument(
        "--resume",
        default=None
    )

    ap.add_argument(
        "--sf",
        type=int,
        default=0
    )

    args = ap.parse_args()

    # Reproducibility
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
        f"  normalisation divisor: {scale}"
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
    # Split by CLEAN IMAGE
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
            len(keys) * args.val_frac
        )
    )

    val_keys = set(
        keys[:n_val_keys]
    )

    tr_idx = [
        i
        for i, (_, _, k) in enumerate(pairs)
        if k not in val_keys
    ]

    va_idx = [
        i
        for i, (_, _, k) in enumerate(pairs)
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
        f"from {len(val_keys)} held-out clean images"
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
        True
    )

    val_ds = PairDataset(
        va_pairs,
        va_sizes,
        args.patch,
        sf,
        scale,
        False
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=SameSizeBatchSampler(
            train_ds,
            args.batch
        ),
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        persistent_workers=args.workers > 0
    )

    val_loader = DataLoader(
        val_ds,
        batch_sampler=SameSizeBatchSampler(
            val_ds,
            max(
                1,
                args.batch // 2
            ),
            shuffle=False,
            drop_last=False
        ),
        num_workers=args.workers
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = NAFNetSR(
        sf=sf,
        width=args.width
    ).to(device)

    # Global pooling during training.
    disable_tlc(model)

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
    # Optimizer and scheduler
    # -----------------------------------------------------------------------

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.9),
        weight_decay=args.weight_decay
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=args.iters,
        eta_min=1e-7
    )

    use_amp = (
        device == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    # -----------------------------------------------------------------------
    # Checkpoints
    # -----------------------------------------------------------------------

    os.makedirs(
        args.out_dir,
        exist_ok=True
    )

    it = 0
    best = -1.0

    if args.resume:

        ck = torch.load(
            args.resume,
            map_location=device
        )

        model.load_state_dict(
            ck["model"]
        )

        opt.load_state_dict(
            ck["opt"]
        )

        sched.load_state_dict(
            ck["sched"]
        )

        it = ck["iter"]

        best = ck.get(
            "best",
            -1.0
        )

        print(
            f"resumed @ {it}, "
            f"best {best:.2f} dB"
        )

    # -----------------------------------------------------------------------
    # Initial validation
    # -----------------------------------------------------------------------

    v, b = evaluate(
        model,
        val_loader,
        device,
        sf
    )

    print(
        f"before training: "
        f"{v:.2f} dB | "
        f"bilinear baseline {b:.2f} dB\n"
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    t0 = time.time()

    rl = 0.0
    rg = 0.0
    rf = 0.0

    while it < args.iters:

        for lr_img, gt_img in train_loader:

            if it >= args.iters:
                break

            lr_img = lr_img.to(
                device,
                non_blocking=True
            )

            gt_img = gt_img.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=use_amp
            ):

                out = model(
                    lr_img
                )

                l1 = F.l1_loss(
                    out,
                    gt_img
                )

                lg = gradient_loss(
                    out,
                    gt_img
                )

                lf = fft_loss(
                    out,
                    gt_img
                )

                loss = (
                    l1
                    + args.lambda_grad * lg
                    + args.lambda_fft * lf
                )

            opt.zero_grad(
                set_to_none=True
            )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                opt
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(
                opt
            )

            scaler.update()

            sched.step()

            rl += l1.item()
            rg += lg.detach().item()
            rf += lf.detach().item()

            it += 1

            # ---------------------------------------------------------------
            # Training log every 100 iterations
            # ---------------------------------------------------------------

            if it % 100 == 0:

                print(
                    f"iter {it:7d}  "
                    f"L1 {rl / 100:.4f}  "
                    f"grad {rg / 100:.4f}  "
                    f"fft {rf / 100:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  "
                    f"{(time.time() - t0) / 100:.3f}s/it"
                )

                rl = 0.0
                rg = 0.0
                rf = 0.0

                t0 = time.time()

            # ---------------------------------------------------------------
            # Validation and checkpoint every eval_every iterations
            # ---------------------------------------------------------------

            if it % args.eval_every == 0:

                v, b = evaluate(
                    model,
                    val_loader,
                    device,
                    sf
                )

                ck = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "iter": it,
                    "best": best,
                    "sf": sf,
                    "scale": scale,
                    "train_patch": args.patch,
                    "cfg": dict(
                        middle_blk_num=12,
                        enc_blk_nums=(
                            2,
                            2,
                            4,
                            8
                        ),
                        dec_blk_nums=(
                            2,
                            2,
                            2,
                            2
                        )
                    ),
                    "args": vars(args)
                }

                flag = ""

                if v > best:

                    best = v
                    ck["best"] = v

                    torch.save(
                        ck,
                        os.path.join(
                            args.out_dir,
                            "best.pth"
                        )
                    )

                    flag = "  *saved*"

                torch.save(
                    ck,
                    os.path.join(
                        args.out_dir,
                        "last.pth"
                    )
                )

                print(
                    f"  >> val {v:.2f} dB  "
                    f"(baseline {b:.2f}, "
                    f"best {best:.2f})"
                    f"{flag}"
                )

                t0 = time.time()

    print(
        f"\ndone. "
        f"best val PSNR {best:.2f} dB"
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
