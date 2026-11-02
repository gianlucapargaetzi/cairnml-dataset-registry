"""Every dataset that is read from a web or Hugging Face source has its metadata in metadata/<name>.json.

The Datasets page shows modalities, labels and split sizes from this file, so it is written once from the adapter's
output (see the README) and checked here: each file parses and names the modalities, labels, samples and splits.
"""

import json
from pathlib import Path

import pytest

from cairnml.datasets.external import read_pin

ROOT = Path(__file__).resolve().parent.parent
PINS = sorted(p.stem for p in (ROOT / "datasets").glob("*.yaml"))
OWN = {name for name in PINS if read_pin(ROOT, name).source == "repo"}


# coco-fashion: its metadata needs the two image archives (about 2 GB) to run the adapter. Until they are generated
# once, the test is expected to fail, and it says so in the report.
KNOWN_MISSING = {"coco-fashion": "metadata needs the image archives: run the adapter once and write the file"}


@pytest.mark.parametrize("name", [
    pytest.param(n, marks=pytest.mark.xfail(reason=KNOWN_MISSING[n], strict=True)) if n in KNOWN_MISSING else n
    for n in PINS if n not in OWN
])
def test_a_public_dataset_has_its_metadata(name):
    path = ROOT / "metadata" / f"{name}.json"
    assert path.is_file(), f"metadata/{name}.json is missing: write it from the adapter (see README)"
    meta = json.loads(path.read_text())
    assert meta["schema_version"] == 1
    assert meta["modalities"] and isinstance(meta["labels"], dict)
    assert meta["samples"] > 0 and sum(meta["splits"].values()) == meta["samples"]
