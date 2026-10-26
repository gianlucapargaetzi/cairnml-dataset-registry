<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/cairnml-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/cairnml-logo.svg">
    <img src="assets/cairnml-logo.svg" alt="CairnML" width="480">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/gianlucapargaetzi/cairnml-dataset-registry/actions/workflows/test.yml"><img src="https://github.com/gianlucapargaetzi/cairnml-dataset-registry/actions/workflows/test.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/gianlucapargaetzi/cairnml-dataset-registry/stargazers"><img src="https://img.shields.io/github/stars/gianlucapargaetzi/cairnml-dataset-registry?style=flat-square&logo=github" alt="GitHub stars"></a>
</p>

# CairnML Dataset Registry

The dataset registry lists public datasets and the adapters that read each one into the CairnML dataset format. A training script asks for a dataset by name, optionally at a commit of this repository, and gets the same samples every time.

## Features

- One pin file per dataset, naming its source (Hugging Face, git, a local folder or a web address) and its adapter.
- Adapters that turn the files of each source into one standard format, with named modalities and typed labels.
- Every run records the source revision, the adapter and the registry commit it used.
- Each adapter is tested on a small sample of its source before its change can reach `main`.

## Layout

| Path | Contents |
|---|---|
| `datasets/<name>.yaml` | One pin file per dataset: `source`, `link`, `adapter`, and optionally `revision` and `overrides` |
| `adapters/<adapter>.py` | One `Adapter` class per file, which reads the source into the standard format |
| `tests/` | One test per adapter, run on a small sample of its source |

## Use a dataset

```python
from cairnml.datasets import Dataset

data = Dataset.get_external("coco-fashion")   # latest main
data = Dataset.get_external("coco-fashion", "<commit>")   # a fixed commit, to rerun a run exactly

for sample in data.samples("train"):
    image = sample.modality("image")   # a path
    boxes = sample.label("bbox2d")     # [{"box": [x1, y1, x2, y2], "label": "..."}, ...]
```

The commit fixes both the pin file, which says where the data comes from, and the adapter code, which says how it is read. Without a commit, the latest commit of `main` is used.

## Add a dataset

1. Add `datasets/<name>.yaml`. For a web source, list the addresses under `link`.
2. Write `adapters/<adapter>.py`. Declare the modalities and labels, then call `writer.add(...)` for each sample. Boxes are `[x1, y1, x2, y2]` in pixels. See `adapters/coco_fashion.py` for a complete example.
3. Copy a small sample of the source into `tests/data/<adapter>/`, laid out the way the source is unpacked. `tests/test_adapters.py` runs every adapter on its sample, and fails for an adapter without one.
4. Open a pull request. The `adapters` workflow must pass before it can be merged.

## Keep main working

`main` is what runs get when they name no commit, so only pull requests that pass `.github/workflows/test.yml` can reach it. The workflow installs the public `cairnml` SDK from its main branch.

The tests run on the samples in `tests/data/`, not on the real sources. They prove that an adapter reads its layout, not that a new upstream revision still has it. When a source changes its layout, a run fails with a message that names the adapter. Add a second adapter and an `overrides` entry in the pin file for the affected revisions:

```yaml
overrides:
  - revisions: ["9b1c4e7"]     # start of the source revision(s) with the new layout
    adapter: coco_fashion_v2
```

Old commits of this repository keep reading old revisions with the old adapter.

## Having problems

1. Check the [open issues](https://github.com/gianlucapargaetzi/cairnml-dataset-registry/issues).
2. If your problem is not listed, [open a new issue](https://github.com/gianlucapargaetzi/cairnml-dataset-registry/issues/new) with the dataset name, the commit you used and the full error message.
