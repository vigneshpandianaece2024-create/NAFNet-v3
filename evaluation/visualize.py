"""
Create visual comparisons for NAFNet v3.

Creates a side-by-side image containing:

    Input LR
    NAFNet v3 output
    Ground Truth

Supported input formats:
    .png
    .jpg
    .jpeg
    .bmp
    .tif
    .tiff
    .npy

Example:

    python evaluation/visualize.py \
        --input results/input \
        --pred results/pred \
        --gt results/gt \
        --output results/comparisons
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {
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
    Load an image as grayscale float32 in [0, 1].
    """

    path = Path(path)

    if path.suffix.lower() == ".npy":

        image = np.load(
            path
        ).astype(
            np.float32
        )

        if image.ndim == 3:

            if image.shape[0] == 1:
                image = image[0]

            elif image.shape[-1] == 1:
                image = image[..., 0]

            else:
                raise ValueError(
                    f"Unsupported array shape: "
                    f"{image.shape}"
                )

    else:

        image = np.asarray(
            Image.open(path).convert("L"),
            dtype=np.float32,
        )

        image /= 255.0

    if image.ndim != 2:

        raise ValueError(
            f"Expected H x W image, "
            f"got {image.shape}"
        )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    if image.max() > 1.5:
        image /= 255.0

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return image


def to_pil(
    image,
    size=None,
):
    """
    Convert float image to PIL grayscale.
    """

    image = (
        image * 255.0
    ).round().astype(
        np.uint8
    )

    result = Image.fromarray(
        image,
        mode="L",
    )

    if size is not None:

        result = result.resize(
            size,
            Image.Resampling.BICUBIC,
        )

    return result


def find_file(
    directory,
    stem,
):
    """
    Find a file with a matching stem.
    """

    directory = Path(
        directory
    )

    for extension in IMAGE_EXTENSIONS:

        candidate = (
            directory
            / f"{stem}{extension}"
        )

        if candidate.exists():

            return candidate

    return None


def add_label(
    image,
    text,
):
    """
    Add a label above an image.
    """

    label_height = 36

    canvas = Image.new(
        "L",
        (
            image.width,
            image.height + label_height,
        ),
        255,
    )

    canvas.paste(
        image,
        (
            0,
            label_height,
        ),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    try:

        font = ImageFont.truetype(
            "DejaVuSans.ttf",
            20,
        )

    except OSError:

        font = ImageFont.load_default()

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    text_width = (
        bbox[2] - bbox[0]
    )

    text_height = (
        bbox[3] - bbox[1]
    )

    x = max(
        0,
        (
            image.width
            - text_width
        ) // 2,
    )

    y = (
        label_height
        - text_height
    ) // 2

    draw.text(
        (x, y),
        text,
        fill=0,
        font=font,
    )

    return canvas


def create_comparison(
    input_image,
    prediction,
    ground_truth,
    output_path,
):
    """
    Create a three-panel comparison.
    """

    # ---------------------------------------------------------------
    # Match dimensions
    # ---------------------------------------------------------------

    target_size = (
        prediction.width,
        prediction.height,
    )

    input_image = input_image.resize(
        target_size,
        Image.Resampling.BICUBIC,
    )

    ground_truth = ground_truth.resize(
        target_size,
        Image.Resampling.BICUBIC,
    )

    input_image = add_label(
        input_image,
        "Input",
    )

    prediction = add_label(
        prediction,
        "NAFNet v3",
    )

    ground_truth = add_label(
        ground_truth,
        "Ground Truth",
    )

    # ---------------------------------------------------------------
    # Create canvas
    # ---------------------------------------------------------------

    width = (
        input_image.width
        + prediction.width
        + ground_truth.width
    )

    height = max(
        input_image.height,
        prediction.height,
        ground_truth.height,
    )

    canvas = Image.new(
        "L",
        (
            width,
            height,
        ),
        255,
    )

    x = 0

    for image in (
        input_image,
        prediction,
        ground_truth,
    ):

        canvas.paste(
            image,
            (
                x,
                0,
            ),
        )

        x += image.width

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create NAFNet v3 "
            "visual comparisons."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input LR image directory.",
    )

    parser.add_argument(
        "--pred",
        required=True,
        help="Prediction image directory.",
    )

    parser.add_argument(
        "--gt",
        required=True,
        help="Ground-truth image directory.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Comparison output directory.",
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input
    )

    pred_dir = Path(
        args.pred
    )

    gt_dir = Path(
        args.gt
    )

    output_dir = Path(
        args.output
    )

    if not input_dir.is_dir():

        raise FileNotFoundError(
            f"Input directory not found: "
            f"{input_dir}"
        )

    if not pred_dir.is_dir():

        raise FileNotFoundError(
            f"Prediction directory not found: "
            f"{pred_dir}"
        )

    if not gt_dir.is_dir():

        raise FileNotFoundError(
            f"Ground-truth directory not found: "
            f"{gt_dir}"
        )

    prediction_files = sorted(
        p
        for p in pred_dir.iterdir()
        if p.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    if not prediction_files:

        raise RuntimeError(
            f"No prediction images found in "
            f"{pred_dir}"
        )

    created = 0
    skipped = 0

    for prediction_path in prediction_files:

        stem = prediction_path.stem

        input_path = find_file(
            input_dir,
            stem,
        )

        gt_path = find_file(
            gt_dir,
            stem,
        )

        if (
            input_path is None
            or gt_path is None
        ):

            skipped += 1

            print(
                f"Skipping {stem}: "
                "matching input/GT not found."
            )

            continue

        input_image = to_pil(
            load_image(
                input_path
            )
        )

        prediction = to_pil(
            load_image(
                prediction_path
            )
        )

        ground_truth = to_pil(
            load_image(
                gt_path
            )
        )

        output_path = (
            output_dir
            / f"{stem}_comparison.png"
        )

        create_comparison(
            input_image,
            prediction,
            ground_truth,
            output_path,
        )

        created += 1

        print(
            f"Created: "
            f"{output_path}"
        )

    print()
    print(
        "=" * 60
    )
    print(
        "NAFNet v3 Visualization"
    )
    print(
        "=" * 60
    )

    print(
        f"Created : {created}"
    )

    print(
        f"Skipped : {skipped}"
    )

    print(
        f"Output  : {output_dir}"
    )


if __name__ == "__main__":
    main()
