"""The checks that keep the default branch working. Every pin file must be valid, every adapter it names must
load, and every adapter must read its own tiny sample (tests/data/<adapter>/) into the standard format.

A new adapter without a sample fails here, so it cannot be merged. Run from the repository root.
"""

from pathlib import Path

import re

import pytest

from cairnml.datasets import SampleWriter
from cairnml.datasets.external import Pin, load_adapter, read_pin

ROOT = Path(__file__).resolve().parent.parent
PINS = sorted(p.stem for p in (ROOT / "datasets").glob("*.yaml"))
OWNERS = {
    line.strip().lower() for line in (ROOT / "allowed-owners.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
}


def adapters_of(pin: Pin):
    return [a for a in [pin.adapter] + [o["adapter"] for o in pin.overrides] if a]


@pytest.mark.parametrize("name", PINS)
def test_the_pin_file_is_valid(name):
    pin = read_pin(ROOT, name)
    assert pin.source in ("huggingface", "git", "local", "repo", "web"), f"{name}: unknown source {pin.source!r}"
    for override in pin.overrides:
        assert override.get("revisions"), f"{name}: an override has no revisions"
        assert override.get("adapter"), f"{name}: an override has no adapter"


@pytest.mark.parametrize("name", PINS)
def test_a_dataset_of_our_own_is_a_pointer_to_a_repository_of_an_allowed_owner(name):
    """How a dataset of our own is read is written in its repository, so its entry holds nothing but the link."""
    pin = read_pin(ROOT, name)
    if pin.source != "repo":
        pytest.skip("not a dataset of our own")
    match = re.fullmatch(r"https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?", pin.link)
    assert match, f"{name}: link must be https://github.com/<owner>/<name>, not {pin.link!r}"
    assert match.group(1).lower() in OWNERS, f"{name}: {match.group(1)} is not in allowed-owners.txt"
    assert not pin.adapter and not pin.overrides, f"{name}: an entry of our own names no adapter; dataset.yaml does"


@pytest.mark.parametrize("name", PINS)
def test_every_adapter_a_pin_file_names_exists(name):
    for adapter in adapters_of(read_pin(ROOT, name)):
        load_adapter(ROOT, adapter)


ADAPTERS = sorted(p.stem for p in (ROOT / "adapters").glob("*.py"))


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_every_adapter_reads_its_sample(adapter, tmp_path):
    sample = ROOT / "tests" / "data" / adapter
    assert sample.is_dir(), f"adapter {adapter!r} needs a tiny sample of its source in tests/data/{adapter}/"
    instance = load_adapter(ROOT, adapter)
    writer = SampleWriter(tmp_path, dict(instance.modalities), dict(instance.labels), source_root=sample)
    instance.run(sample, writer)
    assert writer.close() > 0, f"adapter {adapter!r} found no samples in its sample"


def test_there_is_no_adapter_that_no_pin_file_uses():
    used = {a for name in PINS for a in adapters_of(read_pin(ROOT, name))}
    assert set(ADAPTERS) == used, f"unused adapters: {sorted(set(ADAPTERS) - used)}; missing: {sorted(used - set(ADAPTERS))}"
