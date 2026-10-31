"""CIFAR-10: the Python batches of the publisher's archive, read into the standard format.

The archive unpacks into cifar-10-python/cifar-10-batches-py/: data_batch_1 to data_batch_5 are the train split,
test_batch is the test split, and batches.meta names the classes. Each row of a batch is one 32 by 32 image as
3072 bytes: the red, green and blue planes in turn. Every image is written as a PNG into images/<split>/ in the
source folder, and its label is the class name.
"""

import pickle
import struct
import zlib
from pathlib import Path

from cairnml.datasets import Adapter

FOLDER = Path("cifar-10-python") / "cifar-10-batches-py"
SPLITS = {"train": [f"data_batch_{number}" for number in range(1, 6)], "test": ["test_batch"]}
SIZE = 32


class _Stub:
    """Stands in for a numpy object in a batch: it keeps the state the pickle gives it, and builds nothing."""

    def __init__(self, *args):
        self.args = args
        self.state = None

    def __setstate__(self, state):
        self.state = state


# The publisher pickles the image rows as numpy arrays. Numpy is not needed to read the raw bytes, so only the
# calls that build an array of bytes are accepted; any other object in a batch is refused.
ALLOWED = {
    ("numpy", "ndarray"): _Stub,
    ("numpy", "dtype"): _Stub,
    ("numpy.core.multiarray", "_reconstruct"): _Stub,
    ("numpy._core.multiarray", "_reconstruct"): _Stub,
}


class _Plain(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"the batch refers to {module}.{name}, which a CIFAR-10 batch does not need")
        return ALLOWED[(module, name)]


def _load(path: Path):
    with open(path, "rb") as handle:
        return _Plain(handle, encoding="bytes").load()


def _rows(value) -> list:
    """The rows of a batch's data as bytes: a numpy array of uint8 (rows of 3072 bytes) or a list of bytes."""
    if isinstance(value, _Stub):
        # numpy pickles an array as (version, shape, dtype, is_fortran, raw bytes)
        _, shape, dtype, is_fortran, raw = value.state
        kind = dtype.args[0]
        kind = kind.decode() if isinstance(kind, bytes) else kind
        if kind not in ("u1", "uint8") or is_fortran:
            raise ValueError(f"the batch data is {kind}, not uint8 in C order")
        width = shape[1]
        return [raw[row * width:(row + 1) * width] for row in range(shape[0])]
    return [bytes(row) for row in value]


def _png(path: Path, width: int, height: int, channels: int, pixels: bytes) -> None:
    """A PNG file (8 bits per channel, no filtering) written with the standard library only."""
    stride = width * channels
    raw = b"".join(b"\x00" + pixels[row * stride:(row + 1) * stride] for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    color = 2 if channels == 3 else 0
    header = struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    )


class Cifar10(Adapter):
    modalities = {"image": "image"}
    labels = {"class": "class"}

    def run(self, source: Path, writer) -> None:
        folder = source / FOLDER
        meta = _load(folder / "batches.meta")
        names = [name.decode() if isinstance(name, bytes) else name for name in meta[b"label_names"]]
        plane = SIZE * SIZE
        for split, batches in SPLITS.items():
            out = source / "images" / split
            out.mkdir(parents=True, exist_ok=True)
            for batch in batches:
                data = _load(folder / batch)
                for index, (row, label) in enumerate(zip(_rows(data[b"data"]), data[b"labels"])):
                    if len(row) != 3 * plane:
                        raise ValueError(f"{batch}: row {index} has {len(row)} bytes, expected {3 * plane}")
                    pixels = bytearray(3 * plane)
                    pixels[0::3] = row[0:plane]
                    pixels[1::3] = row[plane:2 * plane]
                    pixels[2::3] = row[2 * plane:]
                    sample_id = f"{batch}-{index:05d}"
                    _png(out / f"{sample_id}.png", SIZE, SIZE, 3, bytes(pixels))
                    writer.add(sample_id, split, {"image": f"images/{split}/{sample_id}.png"},
                               {"class": names[label]})
