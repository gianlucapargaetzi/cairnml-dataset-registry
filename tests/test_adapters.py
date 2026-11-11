"""The checks that keep the default branch working. Every pin file must be valid, every adapter it names must
load, and every adapter must read its own tiny sample (tests/data/<adapter>/) into the standard format.

A new adapter without a sample fails here, so it cannot be merged. Run from the repository root.
"""

import ast
import gzip
import hashlib
import importlib.util
import sysconfig
import zipfile
from pathlib import Path

import re

import pytest

from cairnml.datasets import DatasetError, SampleWriter
from cairnml.datasets.external import Pin, adapter_requirements, load_adapter, load_adapter_module, read_pin

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


@pytest.mark.parametrize("name", PINS)
def test_a_source_that_is_not_local_has_a_fetch_in_its_adapter(name):
    pin = read_pin(ROOT, name)
    if pin.source == "local":
        pytest.skip("a local folder is used where it is")
    _, module = load_adapter_module(ROOT, pin.adapter)
    assert callable(getattr(module, "fetch", None)), f"{name}: adapter {pin.adapter!r} must define fetch for source {pin.source!r}"


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_the_requirements_of_an_adapter_are_exact_pins(adapter):
    _, module = load_adapter_module(ROOT, adapter)
    adapter_requirements(module, adapter)


# The import name of each package that cifar10 may import, and the distribution it is installed from.
DISTRIBUTIONS = {"pyarrow": "pyarrow", "huggingface_hub": "huggingface_hub"}


def _is_standard_library(name):
    """sys.stdlib_module_names exists from Python 3.10 only, and the adapters run on 3.9: ask the import system."""
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False
    origin = spec.origin or ""
    return origin in ("built-in", "frozen") or (
        origin.startswith(sysconfig.get_paths()["stdlib"]) and "-packages" not in origin
    )


def test_cifar10_requires_every_package_it_imports():
    """A package the adapter imports must be named in REQUIREMENTS, so a run that lacks it is refused with the command."""
    tree = ast.parse((ROOT / "adapters" / "cifar10.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    third_party = {name for name in imported if not _is_standard_library(name) and name != "cairnml"}
    _, module = load_adapter_module(ROOT, "cifar10")
    named = {re.match(r"[\w.-]+", requirement).group(0).lower() for requirement in adapter_requirements(module, "cifar10")}
    assert third_party, "the adapter imports no package outside the standard library: the test is out of date"
    for package in third_party:
        assert DISTRIBUTIONS.get(package, package).lower() in named, f"cifar10 imports {package} but REQUIREMENTS does not name it"


def test_coco_fashion_fetches_its_addresses_and_unpacks_the_archives(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "annotations.json").write_text('{"images": []}')
    with zipfile.ZipFile(site / "images.zip", "w") as archive:
        archive.writestr("train/a.jpg", b"a")
    _, module = load_adapter_module(ROOT, "coco_fashion")
    links = [(site / "annotations.json").as_uri(), (site / "images.zip").as_uri()]
    dest = tmp_path / "out"
    module.fetch(links, module.resolve(links, None), dest)
    assert (dest / "annotations.json").read_text() == '{"images": []}'
    assert (dest / "images" / "train" / "a.jpg").read_bytes() == b"a"
    assert not (dest / "images.zip").exists()


def test_coco_fashion_revision_changes_when_a_file_changes(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "annotations.json").write_text("{}")
    _, module = load_adapter_module(ROOT, "coco_fashion")
    links = [(site / "annotations.json").as_uri()]
    first = module.resolve(links, None)
    assert module.resolve(links, "given") == "given"
    assert module.resolve(links, None) == first
    (site / "annotations.json").write_text('{"images": [1]}')  # the size the server reports changes with the file
    assert module.resolve(links, None) != first


def test_fashion_mnist_refuses_a_download_that_does_not_match_its_checksum(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "labels.gz").write_bytes(gzip.compress(b"one file"))
    good = hashlib.md5(gzip.compress(b"one file")).hexdigest()
    _, module = load_adapter_module(ROOT, "fashion_mnist")
    entry = {"url": (site / "labels.gz").as_uri(), "checksum": f"md5:{good}", "unpack": "gz"}
    dest = tmp_path / "out"
    module.fetch([entry], "any", dest)
    assert (dest / "labels").read_bytes() == b"one file"

    entry = {"url": (site / "labels.gz").as_uri(), "checksum": "md5:" + "0" * 32, "unpack": "gz"}
    with pytest.raises(DatasetError, match="does not match its md5 checksum"):
        module.fetch([entry], "any", tmp_path / "out2")
