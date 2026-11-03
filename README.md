# Geometry Optimization and GNN Prediction of Solvation Free Energies

This repository contains the GNN training code used for the manuscript:

*Is Full Geometry Optimization Always Necessary for Graph Neural Network Learning? Insights from a Multi-Stage Quantum Chemical Dataset*

We investigate how molecular geometry optimization affects the prediction of solvation free energies. The study compares 435 solute–solvent systems across eight geometry stages, from the initial structures to the final optimized geometries.

## Models

Two GNN architectures are included:

- **3D-MPNN**: a custom distance-based message-passing model.
- **DimeNet++**: implemented using PyTorch Geometric.

For each architecture, we test three input settings:

- **B**: 3D molecular geometry only.
- **E**: 3D geometry combined with 94 structural descriptors.
- **Z**: the E model with the descriptor inputs set to zero.

## Try the example

A small synthetic dataset is provided in `data/demo/` so that the code can be tested without downloading the original research data. It contains 16 systems, 32 XYZ files (initial and final), and the required CSV files.

The code was tested with Python 3.11, PyTorch 2.10.0, and PyTorch Geometric 2.6.1. Install a PyTorch build suitable for your system before installing the other requirements.

```bash
pip install -r requirements.txt

# Check the example files
python scripts/run_demo.py --inspect

# Run one model (1 epoch, 2 folds)
bash scripts/run_demo.sh

# Run all six model configurations
bash scripts/run_demo.sh all
```

The training outputs are saved in `run_outputs/demo/`.

**Note:** The example data and targets are synthetic. They are provided only to test the code and do not reproduce the results reported in the manuscript.

## Reproducing the study

The full study uses 435 systems at eight geometry stages (`initial`, `loose_t10`, `loose_t20`, `loose_t30`, `tight_t10`, `tight_t20`, `tight_t30`, and `final`). The reported experiments use five-fold random cross-validation and leave-one-solvent-out validation, with three random seeds.

The complete XYZ structures, experimental targets, descriptor table, and original data-split information are **not included** in this repository. These data are needed to reproduce the reported results.

Once the original data are available, the experiment commands can be inspected using:

```bash
python scripts/run_matched.py \
  --manifest /path/to/manifest.csv \
  --descriptors /path/to/descriptors.csv \
  --reference-folds /path/to/reference_outer_folds_B_E.csv \
  --outroot run_outputs/full_experiment --dry-run
```

See `docs/DEMO_DATASETS.md` and `configs/experiment_settings.json` for additional information.

## Repository contents

- `models/`, `experiment_scripts/`: GNN training code.
- `scripts/`: demo, training launchers, and validation scripts.
- `data/demo/`: small synthetic example dataset.
- `analysis/`: prediction checks and statistical analysis scripts.
- `tests/`: basic code and input checks.
- `docs/`: additional methods and data information.

## Availability

The code is provided for research inspection and reproducibility. The full research dataset will be linked here when its release is completed. A software license will be added after institutional authorization; public access alone does not grant reuse rights.
