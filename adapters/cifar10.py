"""CIFAR-10 from its Hugging Face copy: the Parquet files of the repository, read into the standard format.

The snapshot holds plain_text/train-00000-of-00001.parquet and plain_text/test-00000-of-00001.parquet. Each row has
the image as PNG bytes (column img) and the class index (column label). Every image is written as a PNG into
images/<split>/ in the source folder, and its label is the class name.
"""

from pathlib import Path

import pyarrow.parquet as pq

from cairnml.datasets import Adapter

FOLDER = Path("plain_text")
SPLITS = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class Cifar10(Adapter):
    modalities = {"image": "image"}
    labels = {"class": "class"}

    def run(self, source: Path, writer) -> None:
        for split, file in SPLITS.items():
            out = source / "images" / split
            out.mkdir(parents=True, exist_ok=True)
            table = pq.read_table(source / FOLDER / file).to_pydict()
            for index, (item, label) in enumerate(zip(table["img"], table["label"])):
                sample_id = f"{split}-{index:05d}"
                (out / f"{sample_id}.png").write_bytes(item["bytes"])
                writer.add(sample_id, split, {"image": f"images/{split}/{sample_id}.png"},
                           {"class": CLASSES[label]})
