# Geometry optimization and GNN prediction of solvation free energies

This repository provides the author-used training code for a **strictly matched,
435-system, eight-geometry-stage benchmark** of solute–solvent clusters.
We evaluate two model backbones—custom **distance-gated 3D-MPNN** and
**PyTorch Geometric DimeNet++**—with three input variants:

- **B:** graph geometry alone;
- **E:** graph features combined with 94 coordinate-derived descriptors;
- **Z:** E's architecture with the standardized descriptors set to zero.

The manuscript is *Is Full Geometry Optimization Always Necessary for Graph Neural
Network Learning? Insights from a Multi-Stage Quantum Chemical Dataset*.
The released code contains the six primary model runners, configuration snapshots,
analysis utilities, and an explicit training launcher. The reported scores are **not**
obtained from stock SchNet. The PyG backbone module retains an optional SchNet
code path, but no stand-alone SchNet benchmark is claimed here.

## Quick start: runnable code with an included small demo

This repository includes a **portable 16-system synthetic fixture** in
`data/demo/` (32 XYZ coordinates for `initial` and `final`; a 32-row manifest;
a 314-column descriptor CSV). The six original runners are called without
modifying their source files.

```bash
# After installing PyTorch, PyG and RDKit per Installation below:
python scripts/run_demo.py --inspect        # CSV, XYZ and paths: no GPU required
bash scripts/run_demo.sh                  # B3D: 1 epoch, Random 2-fold
bash scripts/run_demo.sh B3D BDim         # both geometry-only backbones
bash scripts/run_demo.sh all              # all six B/E/Z model configurations
bash scripts/check_release.sh             # SHA256, tests and all demo preflights
```

All training outputs are written under Git-ignored `run_outputs/demo/`.
**The bundled demo is entirely synthetic and its labels/descriptors are NOT
experimental, ORCA, or GFN2-xTB results. It tests execution only, never
reproduces paper accuracy.** See [`docs/DEMO_DATASETS.md`](docs/DEMO_DATASETS.md).

The source benchmark data have a separate release process. An author-side
extractor, `scripts/build_real_demo_from_server.py`, can copy a small set of
real XYZ files and **rewrite absolute server paths to portable paths**.
The generated real subset is ignored by Git until institutional and upstream
data redistribution rights are approved. No original experimental labels or
structures are included in the bundled synthetic fixture.

## Experiment definition

Eight **categorical** structure stages are `initial`, `loose_t10`, `loose_t20`,
`loose_t30`, `tight_t10`, `tight_t20`, `tight_t30`, and `final`. The same 435
systems and targets are used in each stage. Each of B/E/Z within each backbone
uses random five-fold and seven leave-one-solvent-out (LOSO) outer splits,
with three seeds `0,1,2`—**288 model fits per configuration**.

All six matched model configurations use these saved run settings:

| Parameter | Value |
|---|---|
| Maximum epochs / early stopping patience | 600 / 100 |
| Batch size; seeds | 16; 0, 1, 2 |
| Random cross-validation folds; LOSO solvents | 5; 7 |
| Validation fraction | 0.15 |
| Hidden width; message/interaction blocks | 96; 4 |
| Upper spatial cutoff | 5.0 Å |
| Learning rate; weight decay | 0.001; 0.00001 |
| Dropout; radial bases (3D-MPNN) | 0.10; 32 |
| DimeNet++ max neighbors | 128 |
| E/Z descriptor branch | 94 → 64 → 32, SiLU, LayerNorm, dropout |

Some original runners have CLI **defaults of 300/50** and abbreviated stages/seeds.
Do **not** run those defaults to reproduce the paper. Use
`scripts/run_matched.py` or explicitly set the manuscript values.

**3D-MPNN**: directed atomic edges use `0.7 < distance < 5.0 Å`;
32 Gaussian radial basis components and sigmoid-gated neighbor messages are
aggregated by **mean**, updated by residual connections and LayerNorm.
The final graph vector concatenates global mean and max node pooling.
This custom model is *related to* continuous-distance GNNs such as SchNet,
not a PyG SchNet model.

**DimeNet++**: separate element and solvent-role embeddings are **added**;
`z + 100*role` is only an API transport code, not a new chemical element.
The feature dimension after its graph readout is **48**, using atom-sum divided
by atom count. The DimeNet++ output initializer is `glorot_orthogonal`.
Native PyTorch radius/triplet routines replace optional extension dependencies;
the neighbor cap of 128 can affect edge enumeration, so an identical upper
cutoff does **not** imply identical graphs between architectures.

## Contents

- `models/3dmpnn/run_3dmpnn_B_original.py`: author-used B trainer.
- `experiment_scripts/run_3dmpnn_E_original.py`, `run_3dmpnn_Z.py`: E/Z trainers.
- `experiment_scripts/run_3dmpnn_Z_worker.py`: original Z stage worker.
- `models/dimenetpp/run_organic_geometry_PyG_{B,E,Z}.py`: DimeNet++ entry points.
- `models/dimenetpp/pyg3d_backbone.py`, `pyg_native_graph_ops.py`,
  `pyg_role_embedding.py`: required shared modules.
- `configs/experiment_settings.json`: **sanitized settings extracted from 51
  original run-config JSON files**, with source hashes; not full run configs.
- `SOURCE_SHA256.txt`: hashes of the **unchanged ten original .py files**.
- `scripts/run_matched.py`: explicit, preflighted training launcher (dry-run by default).
- `analysis/verify_matched_predictions.py`: compare output to the published
  96 pooled RMSE entries; **independently reconstructed analysis**, not an
  original training program.
- `analysis/bootstrap_from_predictions.py`: matched solvent-group bootstrap
  and per-solvent analysis, independently implemented from the stated methods.
- `analysis/check_neighbor_cap.py`: inspect 5 Å neighbor counts for all 3,480
  matched XYZ geometries, including whether the 128-neighbor cap is reached.
- `tests/test_native_ops.py`, `tests/test_neighbor_cap.py`: native graph and
  neighbor-count sanity tests; reference comparisons need `torch-cluster`.
- `scripts/run_demo.py`, `scripts/run_demo.sh`: ready-to-run small demo.
- `scripts/generate_synthetic_demo.py`: recreate the synthetic fixture.
- `scripts/build_real_demo_from_server.py`: opt-in extraction of a small original-data subset.
- `data/demo/`, `data/descriptor_columns.json`: portable synthetic XYZ/CSV example and exact original descriptor field names.
- `docs/DEMO_DATASETS.md`: provenance and optional real-subset packaging instructions.
- `docs/GITHUB_UPLOAD.md`: safe GitHub Private upload commands.

Original source file names/locations are kept to preserve their SHA256 provenance
and the original Z worker's relative import path.

## Installation

The author-observed environment was Python 3.11.8, PyTorch 2.10.0+cu128,
PyG 2.6.1, RDKit 2022.09.5, NumPy 1.26.4, pandas 2.3.2 and scikit-learn 1.7.1.
For a Linux/NVIDIA CUDA 12.8 environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c 'import torch, torch_geometric, rdkit; print(torch.__version__, torch_geometric.__version__, rdkit.__version__)'
```

Install versions appropriate to your operating system and CUDA stack; the
commands above document one environment used for this benchmark. The included
graph-operator tests do not replace the full original-dataset neighbor audit.

## Required scientific data for article reproduction (not packaged here)

The exact stage manifest (435 systems × 8 stages = 3,480 rows), source XYZs,
experimental labels, the 314-column descriptor table and its **94 selected
columns**, and the exact Random/LOSO outer membership table
`reference_outer_folds_B_E.csv` are required.

E/Z runners validate descriptor columns 58:122 and 284:314 (Python slices),
including the boundary names `rg_total`,
`contact_5p0_solvent_heavy_atom_fraction`,
`solvent_molecule_min_dist_to_solute_min`, and
`solvent_molecule_within_5p0_fraction`.
**The original descriptor-generation code is not in this repository**;
regenerating those 314 fields independently is not currently guaranteed.

The expected primary manifest fields are `system_id`, `stage_mode`,
`solute_smiles`, `solvent_smiles`, `solvent`, `n_solvent`, `input_xyz`, `target`.
Rows must preserve the original member identities and input-XYZ mapping.
Paths in original manifests may require a documented remapping to the
public XYZ directory. The same manifest row order and reference split
assignments must be maintained for identical reruns.

**Data publication location:** Figshare DOI and version **to be added after
verification of a public data release**. The reserved identifier alone does
not establish a published dataset.

### Exact experiment command

To inspect the precise commands **without training**:

```bash
python scripts/run_matched.py \
  --manifest /path/to/matched_435_manifest.csv \
  --descriptors /path/to/original_314_column_descriptors.csv \
  --reference-folds /path/to/reference_outer_folds_B_E.csv \
  --outroot /path/to/results --dry-run
```

After downloading the required original data and verifying paths, add
`--execute` to perform all six configurations. Use `--models B3D E3D Z3D
BDim EDim ZDim` or a subset to choose models. Avoid changing dataset row
order or split assignments. Running all configurations is computationally
intensive; the command **does not** execute by default.

### Recalculate the paper results

The local mapping file `analysis/prediction_paths.json` and default local training
output folder `run_outputs/` are excluded from Git tracking because they can
contain machine-specific absolute paths. Do not force-add either path.

`verify_matched_predictions.py` expects **six matched-set prediction folders**
(or six individual CSVs), never the larger Available-set results. See
`analysis/prediction_paths_TEMPLATE.json` for the input JSON format:

```bash
python analysis/verify_matched_predictions.py \
  --inputs analysis/prediction_paths.json \
  --report local_matched_audit.json
python analysis/bootstrap_from_predictions.py \
  --inputs analysis/prediction_paths.json \
  --outdir local_analysis
```

The verifier always checks **435 systems per stage, eight stages, three seeds,
matching target labels and exact outer fold assignments** across all six models.
By default it also enforces ±0.00011 agreement with the article's 96 pooled RMSE
entries (Tables 3–4). Use this **strict mode for archived held-out predictions**.

Fresh GPU retraining can yield different numerical predictions because model
training is not bitwise deterministic. Use `--skip-table-check` on **both**
commands below to verify the matched experimental structure and calculate new
bootstrap summaries without rejecting all numerical differences:

```bash
python analysis/verify_matched_predictions.py \
  --inputs analysis/prediction_paths.json \
  --report run_outputs/new_run_audit.json --skip-table-check
python analysis/bootstrap_from_predictions.py \
  --inputs analysis/prediction_paths.json \
  --outdir run_outputs/new_run_analysis --skip-table-check
```

Skipping the article-table comparison **does not skip** matched-set identity,
label or fold consistency checks. The verifier's report explicitly marks the
article-table comparison as skipped; it is not a claim that the rerun numerically
matches the paper. Full statistical replication needs the same held-out
predictions and the exact original analysis procedure.

The independently implemented bootstrap uses 20,000 shared, paired
seven-solvent resamples at seed 20261019. It uses
`numpy.random.default_rng(seed).integers(0,7,size=(replicates,7))` to generate
all resampling indices at once. This may differ from the original analysis
code's random-number consumption, **even if the seed matches**. Before
publication, compare every reported interval to Supplementary Table S5 using
the actual archived predictions. If they differ, update the manuscript and S5
using one documented algorithm or publish the original interval-generating
code. Seven-group percentile intervals are exploratory.

### Small real-training smoke test on A100

Before making the repository public, execute a **one-epoch, one-stage, one-seed,
two-random-fold** diagnostic on the original matched data. It exercises actual
GPU training and prediction CSV generation for the preserved six runners without
retraining the full article benchmark:

```bash
python scripts/smoke_train.py --manifest /absolute/path/to/matched_manifest.csv \
  --descriptors /absolute/path/to/original_314_descriptors.csv --models B3D BDim
CUDA_VISIBLE_DEVICES=3 python scripts/smoke_train.py \
  --manifest /absolute/path/to/matched_manifest.csv \
  --descriptors /absolute/path/to/original_314_descriptors.csv \
  --models B3D BDim --execute
```

After both pass, repeat with `--models B3D E3D Z3D BDim EDim ZDim`.
**The diagnostic deliberately disables reference-fold verification for DimeNet++
because 2-fold smoke splits differ from the article's exact 5-fold splits.**
Article replication must use `scripts/run_matched.py` and the real reference folds.
See `docs/SMOKE_TEST_A100_KR.md` for the full server procedure.

### Check the DimeNet++ 128-neighbor limit

```bash
python analysis/check_neighbor_cap.py \
  --manifest /path/to/matched_435_manifest.csv \
  --cutoff 5.0 --cap 128 \
  --report-csv run_outputs/neighbor_counts.csv \
  --report-json run_outputs/neighbor_summary.json
```

This counts **other atoms at distances strictly less than 5.0 Å** for every
atomic center, matching the native DimeNet++ radius graph's cutoff and self-edge
convention. It requires all 3,480 matched XYZ paths to be readable and fails
if any exceeds the neighbor cap. It also reports pairs lying very close to the
cutoff, where floating-point differences might change edge membership. If the
published files use remapped paths, repeat `--path-map OLD_PREFIX=NEW_PREFIX`.
The script does not alter the original training code.

## Available-set limitation

The stage-specific Available-set contains 435–557 systems depending on the
stage. This differs from the fixed 435-system Intersection-set. Available-set
metrics are not causal comparisons of optimization stages. Older available-set
3D-MPNN runs with cutoff 4 Å must not be pooled with the 5 Å matched scores.

## Citation and licensing

`CITATION.cff` identifies the associated manuscript and authors. No reuse license is granted by mere public visibility of source files: an OSI-approved
`LICENSE` should be added **after author and KRICT release authorization**.
The code archive should remain **private** until that authorization and the
original descriptor data/outer-fold files are made available to reviewers.
Add the confirmed public repository URL, Figshare data DOI, archive DOI,
version, and release date to the repository metadata after they exist.
Do not treat a reserved but unpublished DOI as a publicly released dataset.
