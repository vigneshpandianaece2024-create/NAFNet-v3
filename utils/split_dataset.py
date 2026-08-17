"""
Create a clean-image-level train/validation split for NAFNet v3.

Important:
    Multiple NoisyLR images may correspond to the same clean GT image.

    Therefore, the split is performed using the clean-image ID rather
    than individual noisy files. This prevents data leakage.

Expected structure:

    combined/
    ├── GT/
    │   ├── 000001.npy
    │   ├── 000002.npy
    │   └── ...
    │
    └── NoisyLR/
        ├── 000001.npy
        ├── 000001_01.npy
        ├── 000001_02.npy
        ├── 000002.npy
        └── ...

Example:

    python utils/split_dataset.py \
        --root /path/to/combined \
        --output splits \
        --val_frac 0.03 \
        --seed 0
"""

import argparse
import json
import random
import re
from pathlib import Path


def clean_image_id(
    filename,
):
    """
    Convert a noisy filename into its clean-image ID.

    Examples:

        000001.npy       -> 000001
        000001_01.npy    -> 000001
        000001_02.npy    -> 000001

    If your dataset uses a different naming convention, adjust this
    function accordingly.
    """

    stem = Path(
        filename
    ).stem

    return re.sub(
        r"_\d+$",
        "",
        stem,
    )


def collect_dataset(
    root,
):
    """
    Collect GT images and all corresponding NoisyLR images.
    """

    root = Path(
        root
    )

    gt_dir = root / "GT"
    noisy_dir = root / "NoisyLR"

    if not gt_dir.is_dir():

        raise FileNotFoundError(
            f"Missing GT directory: "
            f"{gt_dir}"
        )

    if not noisy_dir.is_dir():

        raise FileNotFoundError(
            f"Missing NoisyLR directory: "
            f"{noisy_dir}"
        )

    gt_files = sorted(
        gt_dir.glob("*.npy")
    )

    noisy_files = sorted(
        noisy_dir.glob("*.npy")
    )

    if not gt_files:

        raise RuntimeError(
            f"No GT .npy files found in "
            f"{gt_dir}"
        )

    if not noisy_files:

        raise RuntimeError(
            f"No NoisyLR .npy files found in "
            f"{noisy_dir}"
        )

    gt_by_id = {
        clean_image_id(p.name): p
        for p in gt_files
    }

    pairs_by_id = {}

    unmatched = []

    for noisy_path in noisy_files:

        image_id = clean_image_id(
            noisy_path.name
        )

        if image_id not in gt_by_id:

            unmatched.append(
                noisy_path.name
            )

            continue

        pairs_by_id.setdefault(
            image_id,
            [],
        ).append(
            noisy_path
        )

    if unmatched:

        print(
            f"WARNING: "
            f"{len(unmatched)} NoisyLR files "
            f"could not be matched to GT."
        )

        for name in unmatched[:10]:

            print(
                f"  - {name}"
            )

        if len(unmatched) > 10:

            print(
                f"  ... and "
                f"{len(unmatched) - 10} more"
            )

    return (
        gt_by_id,
        pairs_by_id,
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a clean-image-level "
            "train/validation split."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Dataset root containing "
            "GT/ and NoisyLR/."
        ),
    )

    parser.add_argument(
        "--output",
        default="splits",
        help=(
            "Directory in which split files "
            "will be written."
        ),
    )

    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.03,
        help=(
            "Fraction of clean images used "
            "for validation."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Random seed."
        ),
    )

    args = parser.parse_args()

    if not 0.0 < args.val_frac < 1.0:

        raise ValueError(
            "--val_frac must be between "
            "0 and 1."
        )

    gt_by_id, pairs_by_id = (
        collect_dataset(
            args.root
        )
    )

    clean_ids = sorted(
        pairs_by_id.keys()
    )

    if not clean_ids:

        raise RuntimeError(
            "No valid clean-image groups "
            "were found."
        )

    rng = random.Random(
        args.seed
    )

    shuffled_ids = clean_ids[:]

    rng.shuffle(
        shuffled_ids
    )

    n_val = max(
        1,
        int(
            len(shuffled_ids)
            * args.val_frac
        ),
    )

    val_ids = sorted(
        shuffled_ids[:n_val]
    )

    train_ids = sorted(
        shuffled_ids[n_val:]
    )

    train_set = set(
        train_ids
    )

    val_set = set(
        val_ids
    )

    # ---------------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------------

    overlap = (
        train_set
        & val_set
    )

    if overlap:

        raise RuntimeError(
            "DATA LEAKAGE DETECTED: "
            f"{len(overlap)} clean IDs "
            "appear in both splits."
        )

    # ---------------------------------------------------------------
    # Build pair lists
    # ---------------------------------------------------------------

    def build_records(
        ids,
    ):

        records = []

        for image_id in ids:

            gt_path = gt_by_id[
                image_id
            ]

            for noisy_path in sorted(
                pairs_by_id[
                    image_id
                ]
            ):

                records.append(
                    {
                        "clean_id": image_id,
                        "gt": str(
                            gt_path
                        ),
                        "noisy_lr": str(
                            noisy_path
                        ),
                    }
                )

        return records

    train_records = build_records(
        train_ids
    )

    val_records = build_records(
        val_ids
    )

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_json = (
        output_dir
        / "train.json"
    )

    val_json = (
        output_dir
        / "val.json"
    )

    metadata_json = (
        output_dir
        / "metadata.json"
    )

    with train_json.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            train_records,
            f,
            indent=2,
        )

    with val_json.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            val_records,
            f,
            indent=2,
        )

    metadata = {
        "seed": args.seed,
        "validation_fraction": args.val_frac,
        "clean_images_total": len(
            clean_ids
        ),
        "clean_images_train": len(
            train_ids
        ),
        "clean_images_validation": len(
            val_ids
        ),
        "pairs_train": len(
            train_records
        ),
        "pairs_validation": len(
            val_records
        ),
        "train_clean_ids": train_ids,
        "validation_clean_ids": val_ids,
    }

    with metadata_json.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        "=" * 60
    )
    print(
        "NAFNet v3 Dataset Split"
    )
    print(
        "=" * 60
    )

    print(
        f"Clean images : "
        f"{len(clean_ids)}"
    )

    print(
        f"Train images : "
        f"{len(train_ids)}"
    )

    print(
        f"Val images   : "
        f"{len(val_ids)}"
    )

    print(
        f"Train pairs  : "
        f"{len(train_records)}"
    )

    print(
        f"Val pairs    : "
        f"{len(val_records)}"
    )

    print(
        f"Seed         : "
        f"{args.seed}"
    )

    print()
    print(
        f"Saved: {train_json}"
    )

    print(
        f"Saved: {val_json}"
    )

    print(
        f"Saved: {metadata_json}"
    )


if __name__ == "__main__":
    main()
