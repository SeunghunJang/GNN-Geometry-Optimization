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

## Running the models on your own data

The research dataset (XYZ structures and experimental labels) is available on Figshare (https://doi.org/10.6084/m9.figshare.34754763). The runners in `models/` can be used with these or other data:

- **Manifest (CSV):** one row per system and geometry stage, with the columns of `data/demo/manifest.csv` (`system_id`, `stage_mode`, `input_xyz`, `target`, `solute_smiles`, `solvent_smiles`, `solvent`, `n_solvent`). In each XYZ file, the solute atoms must come first, followed by the solvent molecules; the total atom count is checked against the SMILES.
- **Descriptor table (descriptor-fusion and zero-descriptor control only):** a CSV in the 314-column layout of `data/descriptor_columns.json`. Only columns 58–121 and 284–313 (94 descriptors, listed in Supplementary Table S6 of the article) are used as model inputs. The descriptor-calculation code is not included in this repository.
- **Reference folds (optional):** the DimeNet++ runners accept `--reference-folds none`, in which case folds are generated from the seed.

The settings used in the article were a 4.0 Å cutoff, up to 600 epochs with early-stopping patience 100, batch size 16, AdamW (learning rate 1e-3, weight decay 1e-5), dropout 0.10, an inner validation fraction of 0.15, and seeds 0, 1 and 2. The runners' own defaults differ (for example, a 5.0 Å cutoff), so pass these values explicitly. The exact commands used for the article can be printed without starting training:
```bash
python scripts/run_matched.py --manifest data/demo/manifest.csv --dry-run
```
Because GPU training is not bitwise deterministic, new runs are expected to give similar, but not identical, results.

## Repository contents

- `models/`: 3D-MPNN and DimeNet++ training code.
- `scripts/`: example and full-benchmark launchers.
- `data/demo/`: portable synthetic example data.
- `analysis/`: prediction checks and statistical analysis.
- `tests/`: basic code and input tests.

## Data and code availability

The full research dataset is publicly available on Figshare at https://doi.org/10.6084/m9.figshare.34754763 under the CC BY 4.0 license. It contains 29,268 XYZ structures for 557 solute–solvent systems, together with per-structure metadata, including computed energies and experimental solvation free-energy labels. The code in this repository is released under the MIT License (see `LICENSE`).
