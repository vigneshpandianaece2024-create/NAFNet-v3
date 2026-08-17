"""
Convert NAFNet v3 NumPy images to PNG.

Supports:
    - H x W
    - 1 x H x W
    - H x W x 1

Expected NumPy values:
    - [0, 1]
    - [0, 255]

Examples:

    python utils/npy_to_png.py \
        --input results \
        --output results_png

Or for one file:

    python utils/npy_to_png.py \
        --input results/image_000.npy \
        --output results_png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_npy(path):
    """Load a NumPy array and convert it to H x W."""

    image = np.load(path).astype(
        np.float32
    )

    if image.ndim == 2:
        pass

    elif image.ndim == 3:

        if image.shape[0] == 1:
            image = image[0]

        elif image.shape[-1] == 1:
            image = image[..., 0]

        else:
            raise ValueError(
                f"Unsupported 3D shape: "
                f"{image.shape}"
            )

    else:

        raise ValueError(
            f"Unsupported array shape: "
            f"{image.shape}"
        )

    return image


def to_uint8(image):
    """
    Convert an image to uint8.

    If the maximum value is greater than 1.5,
    assume the image is stored in [0, 255].

    Otherwise assume [0, 1].
    """

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    if image.max() > 1.5:
        image = image / 255.0

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return (
        image * 255.0
    ).round().astype(
        np.uint8
    )


def convert_file(
    input_path,
    output_path,
):
    image = load_npy(
        input_path
    )

    image = to_uint8(
        image
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        image,
        mode="L",
    ).save(
        output_path
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert NAFNet v3 "
            ".npy images to PNG."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input .npy file or directory."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output PNG file or directory."
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    # ---------------------------------------------------------------
    # Single file
    # ---------------------------------------------------------------

    if input_path.is_file():

        if input_path.suffix.lower() != ".npy":

            raise ValueError(
                "Input file must be a .npy file."
            )

        if output_path.suffix.lower() != ".png":

            output_path = (
                output_path
                / f"{input_path.stem}.png"
            )

        convert_file(
            input_path,
            output_path,
        )

        print(
            f"Saved: {output_path}"
        )

        return

    # ---------------------------------------------------------------
    # Directory
    # ---------------------------------------------------------------

    if not input_path.is_dir():

        raise FileNotFoundError(
            f"Input does not exist: "
            f"{input_path}"
        )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        input_path.glob("*.npy")
    )

    if not files:

        raise RuntimeError(
            f"No .npy files found in "
            f"{input_path}"
        )

    converted = 0

    for file in files:

        destination = (
            output_path
            / f"{file.stem}.png"
        )

        convert_file(
            file,
            destination,
        )

        converted += 1

    print(
        f"Converted {converted} "
        f"NumPy files."
    )

    print(
        f"PNG output: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
