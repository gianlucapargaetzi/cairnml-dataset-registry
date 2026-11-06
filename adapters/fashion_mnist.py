"""Fashion-MNIST: the IDX files of the publisher, read into the standard format.

The four files unpacked from the publisher's .gz archives sit in the source folder: train-images-idx3-ubyte and
train-labels-idx1-ubyte for the train split, t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte for the test split.
Each image is 28 by 28 gray levels. It is written as a PNG into images/<split>/ in the source folder, and its label
is the class name.
"""

import gzip
import hashlib
import re
import shutil
import struct
import urllib.request
import zlib
from pathlib import Path

from cairnml.datasets import Adapter, DatasetError

# The files are downloaded with the standard library: no Python package is needed.
REQUIREMENTS = []

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


_CHECKSUM = {"md5": 32, "sha256": 64}  # hex digits of each algorithm


def _entries(link) -> list:
    """The pin file's fetch list: each entry an address, or a mapping with url, checksum (md5:<hex> or sha256:<hex>)
    and unpack (none or gz). The publisher's checksums are checked after each download."""
    items = [link] if isinstance(link, (str, dict)) else list(link)
    entries = []
    for item in items:
        item = {"url": item} if isinstance(item, str) else item
        url, checksum, unpack = item.get("url"), item.get("checksum"), item.get("unpack", "none")
        if not isinstance(url, str) or not re.match(r"(https?|file)://", url):
            raise DatasetError(f"not a web address: {url!r}")
        if checksum is not None:
            algorithm, _, value = str(checksum).partition(":")
            if algorithm not in _CHECKSUM or not re.fullmatch(rf"[0-9a-fA-F]{{{_CHECKSUM[algorithm]}}}", value):
                raise DatasetError(f"the checksum of {url} must be md5:<32 hex digits> or sha256:<64 hex digits>")
            checksum = f"{algorithm}:{value.lower()}"
        if unpack not in ("none", "gz"):
            raise DatasetError(f"unpack of {url} must be none or gz, not {unpack!r}")
        entries.append((url, checksum, unpack))
    return entries


def resolve(link, revision):
    """The revision of the files: the one given, else a hash of their checksums (a file without one is asked of the server)."""
    entries = _entries(link)
    if revision:
        return revision
    digest = hashlib.sha256()
    for url, checksum, _ in entries:
        if checksum:
            digest.update(f"{checksum}\n".encode())
            continue
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as reply:
                etag = reply.headers.get("ETag", "")
                size = reply.headers.get("Content-Length", "")
        except OSError as exc:
            raise DatasetError(f"cannot reach {url}: {exc}") from None
        digest.update(f"{url}\n{etag}\n{size}\n".encode())
    return digest.hexdigest()


def fetch(link, revision: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for url, checksum, unpack in _entries(link):
        target = dest / url.rsplit("/", 1)[-1].split("?", 1)[0]
        try:
            urllib.request.urlretrieve(url, target)
        except OSError as exc:
            raise DatasetError(f"cannot download {url}: {exc}") from None
        if checksum:
            algorithm, _, expected = checksum.partition(":")
            hasher = hashlib.new(algorithm)
            with open(target, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
            if digest != expected:
                target.unlink()
                raise DatasetError(
                    f"{target.name} from {url} does not match its {algorithm} checksum (expected {expected}, got "
                    f"{digest}): the download is refused"
                )
        if unpack == "gz":  # one file, written in place of the archive
            if not target.name.endswith(".gz"):
                raise DatasetError(f"{target.name} is not a .gz file, so it cannot be unpacked as gz")
            with gzip.open(target, "rb") as source, open(target.with_name(target.name[:-3]), "wb") as out:
                shutil.copyfileobj(source, out)
            target.unlink()


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
