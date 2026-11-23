# Effects of Geometry Optimization on GNN-Based Solvation Free Energy Prediction

This repository contains the GNN training code associated with the manuscript:

*Is Full Geometry Optimization Always Necessary for Graph Neural Network Predictions? Insights from a Multi-Stage Quantum Chemical Dataset*

We investigate how the degree of molecular geometry optimization affects the prediction of experimental solvation free energies. The benchmark compares 435 solute–solvent systems at eight geometry stages, from initial to final optimized structures.

## Models

Two GNN architectures are included:

- **3D-MPNN:** a custom distance-based message-passing neural network.
- **DimeNet++:** a directional GNN implemented with PyTorch Geometric.

Each architecture is evaluated with three input configurations:

- **Geometry-only:** uses atomic identities, solute/solvent roles, and 3D coordinates, without additional handcrafted descriptors.
- **Descriptor-fusion:** combines the learned 3D graph representation with 94 coordinate-derived descriptors.
- **Zero-descriptor control:** uses the same fusion architecture but sets the descriptor inputs to zero.

The code retains the original abbreviations **B**, **E**, and **Z**, respectively, in script names and model options.

## Run the example

A small synthetic dataset is included in `data/demo/`. It contains 16 systems, 32 XYZ files (initial and final geometries), and the CSV files needed to test all six model configurations.

The code has been tested with Python 3.11, PyTorch 2.10.0, and PyTorch Geometric 2.6.1. Install a PyTorch build appropriate for your system, then install the remaining requirements:

```bash
# Install dependencies
pip install -r requirements.txt

# Check the example inputs without training
python scripts/run_demo.py --inspect

# Run the default model (1 epoch, 2 random folds)
bash scripts/run_demo.sh

# Run all six model configurations
bash scripts/run_demo.sh all
```

Outputs are saved under `run_outputs/demo/`.

**Note:** The example structures, descriptors, and targets are synthetic. They are intended only to verify that the code runs; they do not reproduce the results in the manuscript.

## Reproducing the study

The full benchmark uses 435 systems at eight stages (`initial`, `loose_t10`, `loose_t20`, `loose_t30`, `tight_t10`, `tight_t20`, `tight_t30`, and `final`). Evaluation uses random five-fold cross-validation and leave-one-solvent-out validation with three random seeds.

The original XYZ structures, experimental targets, descriptor table, and reference split assignments are **not included** in this repository. They are required to reproduce the reported results.

When those files are available, the full-experiment commands can be inspected without starting training:

```bash
python scripts/run_matched.py \
  --manifest /path/to/manifest.csv \
  --descriptors /path/to/descriptors.csv \
  --reference-folds /path/to/reference_outer_folds_B_E.csv \
  --outroot run_outputs/full_experiment \
  --dry-run
```

## Repository contents

- `models/`: 3D-MPNN and DimeNet++ training code.
- `scripts/`: example and full-benchmark launchers.
- `data/demo/`: portable synthetic example data.
- `analysis/`: prediction checks and statistical analysis.
- `tests/`: basic code and input tests.

## Data and code availability

The full research dataset is publicly available on Figshare at https://doi.org/10.6084/m9.figshare.34754763. It contains 29,268 XYZ structures for 557 solute–solvent systems, together with per-structure metadata, including computed energies and experimental solvation free-energy labels. The code is currently provided without a software license, pending institutional authorization. Public visibility alone does not grant permission to reuse or redistribute the code.
