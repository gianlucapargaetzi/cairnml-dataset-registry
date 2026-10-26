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

# cairnml-dataset-registry

Public datasets, and the adapters that read each one into the CairnML dataset format. A training script asks for a
dataset by name, and optionally a commit of this repository:

```python
from cairnml.datasets import Dataset
data = Dataset.get_external("coco-fashion")                  # latest main
data = Dataset.get_external("coco-fashion", "<commit>")      # a fixed commit
```

The commit fixes both the pin file (where the data comes from) and the adapter code (how it is read). The run
records the source revision it used, the adapter and this repository's commit.

## Layout

| Path | What |
|---|---|
| `datasets/<name>.yaml` | one pin file per dataset: `source` (`huggingface`, `git`, `local`, `web`), `link`, `adapter`, optional `revision` and `overrides` |
| `adapters/<adapter>.py` | one `Adapter` class per file, turning the files of the source into the standard format |
| `tests/` | one test per adapter on a tiny sample of its source |

## Adding a dataset

1. Add `datasets/<name>.yaml`.
2. Write `adapters/<adapter>.py`. Declare the modalities and labels, then call `writer.add(...)` for each sample.
   Boxes are `[x1, y1, x2, y2]` in pixels. See `adapters/coco_fashion.py`.
3. Copy a tiny sample of the source, laid out the way the source is unpacked, into `tests/data/<adapter>/`.
   `tests/test_adapters.py` runs every adapter on its sample and fails for an adapter without one.
4. Open a pull request. The `adapters` workflow must pass before it can be merged.

## Keeping `main` working

`main` is what runs get when they name no commit, so only pull requests that pass `.github/workflows/test.yml`
may reach it. The workflow installs the public `cairnml` SDK from its main branch.

The tests run on the samples in `tests/data/`, not on the real sources, so they prove that an adapter reads its
layout, not that a new upstream revision still has it. When a source changes its layout, a run fails with a message
that names the adapter. Add a second adapter and an `overrides` entry in the pin file for the affected revisions:

```yaml
overrides:
  - revisions: ["9b1c4e7"]     # start of the source revision(s) with the new layout
    adapter: coco_fashion_v2
```

Old commits of this repository keep reading old revisions with the old adapter.
