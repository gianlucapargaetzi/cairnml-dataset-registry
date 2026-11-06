"""COCO Fashion (Fashionpedia): the COCO-format annotations of the web sources, read into the standard format.

The source folder holds the annotation file of each split and the images unpacked from the archives:
train2020/train/ and val_test2020/test/. Boxes arrive as COCO [x, y, width, height] and are stored as
[x1, y1, x2, y2]. Each split keeps the order of its annotation file.
"""

import hashlib
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from cairnml.datasets import Adapter, DatasetError

# The downloads are plain files and zip archives: no Python package is needed.
REQUIREMENTS = []

SPLITS = {
    "train": ("instances_attributes_train2020.json", "train2020/train"),
    "val": ("instances_attributes_val2020.json", "val_test2020/test"),
}


def _addresses(link) -> list:
    return [link] if isinstance(link, str) else list(link)


def resolve(link, revision):
    """The revision of the files: the one given, else a hash of each address with the ETag and size the server reports."""
    if revision:
        return revision
    digest = hashlib.sha256()
    for url in _addresses(link):
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=60) as reply:
                etag = reply.headers.get("ETag", "")
                size = reply.headers.get("Content-Length", "")
        except OSError as exc:
            raise DatasetError(f"cannot reach {url}: {exc}") from None
        digest.update(f"{url}\n{etag}\n{size}\n".encode())
    return digest.hexdigest()


def fetch(link, revision: str, dest: Path) -> None:
    """Each address is downloaded into dest; a zip archive is unpacked into a folder named after it and removed."""
    dest.mkdir(parents=True, exist_ok=True)
    for url in _addresses(link):
        target = dest / url.rsplit("/", 1)[-1].split("?", 1)[0]
        try:
            urllib.request.urlretrieve(url, target)
        except OSError as exc:
            raise DatasetError(f"cannot download {url}: {exc}") from None
        if target.name.endswith(".zip"):
            with zipfile.ZipFile(target) as archive:
                archive.extractall(dest / target.name[: -len(".zip")])
            target.unlink()


class CocoFashion(Adapter):
    modalities = {"image": "image"}
    labels = {"bbox2d": "bbox2d"}

    def run(self, source: Path, writer) -> None:
        for split, (annotations, folder) in SPLITS.items():
            with open(source / annotations) as handle:
                data = json.load(handle)
            names = {category["id"]: category["name"] for category in data["categories"]}
            boxes = defaultdict(list)
            for ann in data["annotations"]:
                x, y, width, height = ann["bbox"]
                if width <= 0 or height <= 0:  # the source has a few empty boxes ([0, 0, 0, 0]); they are not objects
                    continue
                boxes[ann["image_id"]].append({"box": [x, y, x + width, y + height], "label": names[ann["category_id"]]})
            for image in data["images"]:
                writer.add(str(image["id"]), split, {"image": f"{folder}/{image['file_name']}"},
                           {"bbox2d": boxes[image["id"]]})
