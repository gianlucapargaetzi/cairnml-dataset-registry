"""Fashion-MNIST: the IDX files of the publisher, read into the standard format.

The four files unpacked from the publisher's .gz archives sit in the source folder: train-images-idx3-ubyte and
train-labels-idx1-ubyte for the train split, t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte for the test split.
Each image is 28 by 28 gray levels. It is written as a PNG into images/<split>/ in the source folder, and its label
is the class name.
"""

import struct
import zlib
from pathlib import Path

from cairnml.datasets import Adapter

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
SPLITS = {
    "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}
IMAGE_MAGIC = 2051  # 0x00000803: unsigned bytes, 3 dimensions
LABEL_MAGIC = 2049  # 0x00000801: unsigned bytes, 1 dimension


def _png(path: Path, width: int, height: int, pixels: bytes) -> None:
    """A gray PNG file (8 bits, no filtering) written with the standard library only."""
    raw = b"".join(b"\x00" + pixels[row * width:(row + 1) * width] for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    )


def _images(path: Path):
    data = path.read_bytes()
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != IMAGE_MAGIC or len(data) != 16 + count * rows * cols:
        raise ValueError(f"{path.name} is not an IDX image file of {count} images")
    return data[16:], count, rows, cols


def _labels(path: Path, count: int):
    data = path.read_bytes()
    magic, found = struct.unpack(">II", data[:8])
    if magic != LABEL_MAGIC or found != count or len(data) != 8 + count:
        raise ValueError(f"{path.name} is not an IDX label file of {count} labels")
    return data[8:]


class FashionMnist(Adapter):
    modalities = {"image": "image"}
    labels = {"class": "class"}

    def run(self, source: Path, writer) -> None:
        for split, (image_file, label_file) in SPLITS.items():
            pixels, count, rows, cols = _images(source / image_file)
            labels = _labels(source / label_file, count)
            out = source / "images" / split
            out.mkdir(parents=True, exist_ok=True)
            size = rows * cols
            for index in range(count):
                sample_id = f"{split}-{index:05d}"
                _png(out / f"{sample_id}.png", cols, rows, pixels[index * size:(index + 1) * size])
                writer.add(sample_id, split, {"image": f"images/{split}/{sample_id}.png"},
                           {"class": CLASSES[labels[index]]})
