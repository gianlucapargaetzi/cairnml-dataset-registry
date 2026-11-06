"""CIFAR-10 from its Hugging Face copy: the Parquet files of the repository, read into the standard format.

The snapshot holds plain_text/train-00000-of-00001.parquet and plain_text/test-00000-of-00001.parquet. Each row has
the image as PNG bytes (column img) and the class index (column label). Every image is written as a PNG into
images/<split>/ in the source folder, and its label is the class name.

The adapter also fetches its source: ``resolve`` turns the pin file's revision into a commit of the Hugging Face
dataset, and ``fetch`` downloads that commit into the cache.
"""

import re
from pathlib import Path

from cairnml.datasets import Adapter, DatasetError

# The exact versions the SDK installed for this adapter: the run refuses to start unless these are installed.
REQUIREMENTS = ["huggingface_hub==1.7.1", "pyarrow==23.0.1"]

FOLDER = Path("plain_text")
SPLITS = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
_SHA = re.compile(r"[0-9a-f]{40}")


def _repo_id(link: str) -> str:
    link = re.sub(r"^(hf://datasets/|https?://huggingface\.co/datasets/)", "", link.strip()).strip("/")
    if not re.fullmatch(r"[\w.-]+(/[\w.-]+)?", link):
        raise DatasetError(f"not a Hugging Face dataset: {link!r} (use for example hf://datasets/org/name)")
    return link


def _hub():
    try:
        import huggingface_hub
    except ImportError:
        raise DatasetError("huggingface_hub is not installed: pip install huggingface_hub") from None
    return huggingface_hub


def resolve(link: str, revision):
    """The commit of the dataset repository: the revision itself when it is a full sha, else the one the Hub reports."""
    if revision and _SHA.fullmatch(revision):
        return revision
    return _hub().HfApi().dataset_info(_repo_id(link), revision=revision).sha


def fetch(link: str, revision: str, dest: Path) -> None:
    _hub().snapshot_download(repo_id=_repo_id(link), repo_type="dataset", revision=revision, local_dir=str(dest))


class Cifar10(Adapter):
    modalities = {"image": "image"}
    labels = {"class": "class"}

    def run(self, source: Path, writer) -> None:
        import pyarrow.parquet as pq

        for split, file in SPLITS.items():
            out = source / "images" / split
            out.mkdir(parents=True, exist_ok=True)
            table = pq.read_table(source / FOLDER / file).to_pydict()
            for index, (item, label) in enumerate(zip(table["img"], table["label"])):
                sample_id = f"{split}-{index:05d}"
                (out / f"{sample_id}.png").write_bytes(item["bytes"])
                writer.add(sample_id, split, {"image": f"images/{split}/{sample_id}.png"},
                           {"class": CLASSES[label]})
