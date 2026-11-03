# Demonstration datasets and provenance

## Ready-to-run fixture in `data/demo/`

The bundled `data/demo/` folder contains:

- 16 synthetic molecular systems in three named organic solvents;
- two input stage labels (`initial`, `final`) per system (32 XYZ files);
- a 32-row manifest, with portable paths such as `data/demo/xyz/toy_001_initial.xyz`;
- a schema-compatible, **314-column synthetic descriptor table** including the 94 input positions used by E/Z;
- fabricated demonstration targets that remain constant between the two stages of a system.

**These are synthetic inputs.** They are not experimental measurements, are not
obtained by the original GFN2-xTB/B3LYP workflow, and cannot be used to infer
scientific performance or to reproduce any table of the article. Most of the
314 descriptor values are deliberately simple placeholders with the exact
column names and positions required by the original training scripts.

Use these data only to confirm input parsing, model initialization, short
training, prediction file writing, and B/E/Z input plumbing.

```
python scripts/run_demo.py --inspect
bash scripts/run_demo.sh                  # B3D, 1 epoch, 2 folds, seed 0
bash scripts/run_demo.sh B3D BDim         # 2 architectures, geometry-only
bash scripts/run_demo.sh all              # all 6 B/E/Z configurations
python scripts/run_demo.py --stage final --models B3D BDim --execute
```

Training outputs and machine-specific configuration JSON files are saved under
`run_outputs/demo/` and automatically excluded by `.gitignore`.

## Prepare an OPTIONAL real-data subset from the author server

The actual Intersection-set manifest has **3,480 rows and 8 columns**:
`system_id`, `stage_mode`, `input_xyz`, `target`, `solute_smiles`,
`solvent_smiles`, `solvent`, `n_solvent`.

The descriptor file has **3,480 rows and 314 columns**. Its 94 selected
model descriptors are columns 58:122 and 284:314 (Python zero-based slices).
The source XYZ paths are absolute A100-server paths and are **not portable**.

The extractor below uses only the original author server files. It checks
system/stage identity, exact descriptor header, label/XYZ correspondence,
atom counts, and all XYZ files. It then copies the selected structures,
rewrites both CSV path columns to `data/demo_real/xyz/...`, and produces
portable data, WITHOUT publishing or changing the original CSVs:

```
BASE=/home/solvation_mrl_project/baseline_mrl/strict_experiments
python scripts/build_real_demo_from_server.py \
  --manifest "$BASE/organic_GNN_intersection_manifest.csv" \
  --descriptors "$BASE/organic_contact_descriptors_intersection_all8.csv" \
  --systems 14 --stages initial,final

python scripts/run_demo.py --data-dir data/demo_real --inspect
python scripts/run_demo.py --data-dir data/demo_real --models B3D BDim --execute
```

`data/demo_real/` is **Git-ignored intentionally**; this preparation step
is suitable for internal validation before deciding whether redistribution
is authorized.

After approval by data owners and the institution, explicitly replace the
synthetic public fixture using the same source inputs with:

```
python scripts/build_real_demo_from_server.py \
  --manifest "$BASE/organic_GNN_intersection_manifest.csv" \
  --descriptors "$BASE/organic_contact_descriptors_intersection_all8.csv" \
  --output data/demo --systems 14 --stages initial,final \
  --replace-synthetic --confirm-public-release
```

The public fixture is overwritten only when the current folder contains the
synthetic marker and **both explicit consent flags are passed**. Review the
new manifest, all labels, source terms of use, and Git changes before pushing.
The demonstration retains only a small subset and does not include the
original benchmark's exact outer folds or archived predictions.

## Full article reproduction

The full published benchmark uses 435 systems at all eight stages, 3 seeds,
Random 5-fold and LOSO, and exact reference outer fold assignments. Use
`scripts/run_matched.py` and obtain the official XYZ, descriptor table, labels,
and fold file from the final data archive when released. The small demo and
its 2-fold output are never substitutes for the full experiment.
