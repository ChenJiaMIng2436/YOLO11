# Improved YOLO11 for Fire and Smoke Detection

This repository contains the research implementation of an improved YOLO11 detector for fire and smoke detection. The released model configuration combines a CS-Biformer backbone, AFPN-E, and Shuffle Attention.

## Model configuration

The main model configuration is:

```text
ultralytics/cfg_yolo11/YOLO11/yolo11-CS-Biformer-AFPN-SA.yaml
```

The principal custom components are implemented in:

- `ultralytics/nn/core11/biformer.py`
- `ultralytics/nn/core11/AFPN.py`
- `ultralytics/nn/core11/Attention/sa.py`
- `ultralytics/nn/tasks.py`

In the paper and diagrams, the module is written as **CS-Biformer**. In Python and YAML module identifiers, it is written as `CS_Biformer` because a hyphen is not valid in a Python identifier.

## Installation

Python 3.10 and a CUDA-enabled PyTorch environment are recommended.

```bash
git clone <REPOSITORY_URL>
cd YOLO11
pip install -e .
```

Install a PyTorch build compatible with the local CUDA driver before running training if the default dependency resolution does not match the system.

## Dataset

The repository includes the fire/smoke dataset used by the default configuration. It is organized in YOLO detection format:

```text
YOLO11/
|-- train/
|   |-- images/
|   `-- labels/
|-- valid/
|   |-- images/
|   `-- labels/
|-- test/
|   |-- images/
|   `-- labels/
`-- data.yaml
```

The included `data.yaml` defines two classes:

```text
0: Fire
1: Smoke
```

The included split contains 4,000 training images, 500 validation images, and 500 test images. Every image has a corresponding YOLO-format label file.

If another dataset is stored elsewhere, edit the paths in `data.yaml` or pass another dataset YAML through `--data`.

## Training

Run the default experiment from the repository root:

```bash
python train_yolo11.py
```

Common options can be overridden from the command line:

```bash
python train_yolo11.py \
  --data /path/to/data.yaml \
  --epochs 300 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

To initialize from a compatible checkpoint:

```bash
python train_yolo11.py --weights /path/to/checkpoint.pt
```

Training outputs are written by Ultralytics under `runs/` and are excluded from version control.

## Reproducibility notes

- The default seed is `0` and deterministic training is requested.
- Some CUDA adaptive-pooling backward operations do not have deterministic implementations in some PyTorch versions. PyTorch may emit warnings and small run-to-run numerical differences can remain.
- Record the GPU model, CUDA version, PyTorch version, dataset split, and command-line options when reporting experiments.

## Weights

Trained weights are not included in the source repository. They can be distributed separately through a GitHub Release or a research-archive service after the release files have been verified.

## Upstream project and license

This code is based on the Ultralytics YOLO codebase and retains its GNU Affero General Public License v3.0. See `LICENSE` for the complete license text. Users of this repository must comply with the upstream license and applicable third-party licenses.

## Citation

Paper-specific citation metadata will be added after publication.
