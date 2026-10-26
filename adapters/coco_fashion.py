"""COCO Fashion (Fashionpedia): the COCO-format annotations of the web sources, read into the standard format.

The source folder holds the annotation file of each split and the images unpacked from the archives:
train2020/train/ and val_test2020/test/. Boxes arrive as COCO [x, y, width, height] and are stored as
[x1, y1, x2, y2]. Each split keeps the order of its annotation file.
"""

import json
from collections import defaultdict
from pathlib import Path

from cairnml.datasets import Adapter

SPLITS = {
    "train": ("instances_attributes_train2020.json", "train2020/train"),
    "val": ("instances_attributes_val2020.json", "val_test2020/test"),
}


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
                boxes[ann["image_id"]].append({"box": [x, y, x + width, y + height], "label": names[ann["category_id"]]})
            for image in data["images"]:
                writer.add(str(image["id"]), split, {"image": f"{folder}/{image['file_name']}"},
                           {"bbox2d": boxes[image["id"]]})
