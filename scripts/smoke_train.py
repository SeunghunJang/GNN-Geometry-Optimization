#!/usr/bin/env python3
"""Run a minimal *real* training smoke test of the six preserved GNN runners.

This checks one geometry stage, one seed, two random folds, and one epoch.
It is NOT the published benchmark; the original 10 training source files
remain unchanged. A separate reference-fold check is required for replication.

Dry-run by default; --execute is the explicit training opt-in.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd

from run_matched import RUNNERS, STAGES

REQUIRED_COLUMNS = {'system_id', 'stage_mode', 'solute_smiles', 'solvent_smiles',
                    'solvent', 'n_solvent', 'input_xyz', 'target'}
DESCRIPTOR_SCHEMA = {58: 'rg_total', 121: 'contact_5p0_solvent_heavy_atom_fraction',
                     284: 'solvent_molecule_min_dist_to_solute_min',
                     313: 'solvent_molecule_within_5p0_fraction'}


def preflight(manifest: Path, descriptors: Path, models: list[str], stage: str) -> dict:
    if not manifest.is_file():
        raise ValueError(f'Expected original matched manifest: {manifest}')
    df = pd.read_csv(manifest, low_memory=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'Manifest columns missing: {sorted(missing)}')
    if len(df) != 3480:
        raise ValueError(f'Matched manifest must have 3,480 rows, found {len(df)}; do not use Available-set')
    if df.duplicated(['system_id', 'stage_mode']).any():
        raise ValueError('Repeated (system_id, stage_mode) keys in manifest')
    initial = set(df[df.stage_mode == 'initial'].system_id.astype(str))
    if len(initial) != 435:
        raise ValueError(f'Expected 435 unique systems in initial, found {len(initial)}')
    for st in STAGES:
        group = df[df.stage_mode == st]
        if len(group) != 435 or set(group.system_id.astype(str)) != initial:
            raise ValueError(f'Incorrect matched membership for stage {st}')
    sample = df[df.stage_mode == stage].copy()
    paths = sample.input_xyz.astype(str).tolist()
    nonabsolute = [p for p in paths if not Path(p).is_absolute()]
    if nonabsolute:
        raise ValueError(f'Use absolute source XYZ paths in manifest, e.g. {nonabsolute[0]}')
    unavailable = [p for p in paths if not Path(p).is_file()]
    if unavailable:
        raise ValueError(f'{len(unavailable)} missing XYZ files in stage {stage}; first: {unavailable[0]}')
    need_desc = any(m in {'E3D','Z3D','EDim','ZDim'} for m in models)
    if need_desc:
        if not descriptors.is_file():
            raise ValueError(f'Original 314-column descriptors missing: {descriptors}')
        cols = list(pd.read_csv(descriptors, nrows=0).columns)
        if len(cols) != 314:
            raise ValueError(f'Descriptor CSV has {len(cols)} columns; expected 314')
        for i, field in DESCRIPTOR_SCHEMA.items():
            if cols[i] != field:
                raise ValueError(f'Descriptor schema mismatch at {i}: {cols[i]}, expected {field}')
    return {'matched_manifest_rows': len(df), 'selected_stage': stage,
            'stage_systems': len(sample), 'xyz_paths_exist': True,
            'descriptor_header_validated': need_desc}


def command(model: str, root: Path, manifest: Path, descriptors: Path,
            outdir: Path, stage: str, epochs: int, seed: int) -> list[str]:
    cmd = [sys.executable, '-u', str(root / RUNNERS[model]),
           '--manifest', str(manifest), '--outdir', str(outdir),
           '--stages', stage, '--splits', 'random', '--seeds', str(seed),
           '--n-folds', '2', '--epochs', str(epochs), '--patience', '1',
           '--val-frac', '0.15', '--batch-size', '16',
           '--hidden-dim', '96', '--num-layers', '4', '--n-rbf', '32',
           '--cutoff', '5.0', '--dropout', '0.10',
           '--lr', '0.001', '--weight-decay', '0.00001']
    if model in {'E3D','Z3D','EDim','ZDim'}:
        cmd += ['--descriptors', str(descriptors), '--desc-hidden', '32']
    if model.endswith('Dim'):
        cmd += ['--architecture', 'dimenetpp', '--max-num-neighbors', '128',
                '--role-offset', '100', '--dimenet-radial', '6',
                '--dimenet-spherical', '7', '--dimenet-int-emb', '64',
                '--dimenet-basis-emb', '8', '--dimenet-out-emb', '128',
                # The 2-fold diagnostic CANNOT match the published 5-fold reference.
                # For the original 5-fold, three-seed experiment, use run_matched.py.
                '--reference-folds', 'none']
    return cmd


def check_result(outdir: Path, stage: str, seed: int, folds: int = 2) -> dict:
    prediction = outdir / 'organic_geometry_GNN_intersection_predictions.csv'
    results = outdir / 'organic_geometry_GNN_intersection_results.csv'
    config = outdir / 'organic_geometry_GNN_intersection_config.json'
    for path in (prediction, results, config):
        if not path.is_file():
            raise RuntimeError(f'Missing training output: {path}')
    df = pd.read_csv(prediction)
    needed = {'system_id','stage_mode','split','seed','fold','y_true','y_pred'}
    if not needed.issubset(df.columns):
        raise RuntimeError(f'Prediction columns missing: {sorted(needed - set(df.columns))}')
    if len(df) != 435 or df.system_id.astype(str).nunique() != 435:
        raise RuntimeError(f'Expected one prediction for each of 435 systems; got {len(df)}')
    if df.duplicated('system_id').any():
        raise RuntimeError('Repeated held-out prediction for a system in diagnostic')
    if set(df.stage_mode.astype(str)) != {stage} or set(df.split.astype(str)) != {'random'}:
        raise RuntimeError('Wrong stage/split in diagnostic prediction CSV')
    if set(pd.to_numeric(df.seed)) != {seed} or set(pd.to_numeric(df.fold)) != set(range(folds)):
        raise RuntimeError('Wrong seed or fold numbers in diagnostic prediction CSV')
    if not np.isfinite(df[['y_true','y_pred']].to_numpy(dtype=float)).all():
        raise RuntimeError('NaN/inf found in held-out diagnostic predictions')
    res = pd.read_csv(results)
    if len(res) != folds:
        raise RuntimeError(f'Expected {folds} summary rows, found {len(res)}')
    cfg = json.loads(config.read_text())
    return {'passed': True, 'prediction_rows': len(df), 'folds': sorted(df.fold.unique().tolist()),
            'epoch_limit': cfg.get('epochs'), 'prediction_file': str(prediction)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path,
                        help='Full 3,480-row matched-set manifest, with existing absolute XYZ paths')
    parser.add_argument('--descriptors', type=Path, default=Path('MISSING_314_COLUMNS.csv'),
                        help='Required for E/Z models; exact 314-column original CSV')
    parser.add_argument('--models', nargs='+', choices=list(RUNNERS), default=list(RUNNERS),
                        help='Run B3D BDim first, then all six')
    parser.add_argument('--stage', choices=STAGES, default='initial')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--outroot', type=Path, default=Path('run_outputs/smoke_training'))
    parser.add_argument('--execute', action='store_true',
                        help='Actually train for 1 epoch; otherwise print commands only')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be >= 1')
    root = Path(__file__).resolve().parents[1]
    manifest = args.manifest.expanduser().resolve()
    descriptors = args.descriptors.expanduser().resolve()
    outroot = args.outroot.expanduser().resolve()
    if args.execute:
        try:
            import torch
            import torch_geometric
        except ImportError as e:
            parser.error('Missing dependency in active venv: ' + str(e))
        if not torch.cuda.is_available():
            parser.error('GPU not detected. Run this integration smoke test on the A100 server.')
        print(f'Torch={torch.__version__}; PyG={torch_geometric.__version__}; '
              f'GPU={torch.cuda.get_device_name(torch.cuda.current_device())}', flush=True)
        try:
            info = preflight(manifest, descriptors, args.models, args.stage)
        except ValueError as e:
            parser.error(str(e))
        print('PREFLIGHT PASS:', json.dumps(info), flush=True)

    summary = {'kind': 'one-stage, one-seed, two-fold small real training smoke test',
               'article_replication': False, 'stage': args.stage, 'seed': args.seed,
               'epochs': args.epochs, 'models': {}}
    for model in args.models:
        outdir = outroot / model
        cmd = command(model, root, manifest, descriptors, outdir, args.stage, args.epochs, args.seed)
        print(f'\n[{model}] {shlex.join(cmd)}', flush=True)
        if not args.execute:
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        logfile = outdir / 'smoke.log'
        print(f'RUNNING {model}; detailed log: {logfile}', flush=True)
        with logfile.open('w', encoding='utf-8') as stream:
            result = subprocess.run(cmd, cwd=root, env=os.environ.copy(),
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    check=False)
        if result.returncode:
            print(f'FAIL {model} (exit={result.returncode}); last log lines:', file=sys.stderr)
            print('\n'.join(logfile.read_text(errors='replace').splitlines()[-35:]),file=sys.stderr)
            return result.returncode
        try:
            summary['models'][model] = check_result(outdir, args.stage, args.seed)
        except (ValueError, RuntimeError) as e:
            print(f'FAIL {model} result audit: {e}', file=sys.stderr)
            return 2
        print(f'PASS {model}: 435 held-out predictions; 2 random folds; '
              f'{args.epochs} epoch(s) maximum',flush=True)
    if args.execute:
        outroot.mkdir(parents=True, exist_ok=True)
        target = outroot / 'SMOKE_PASS_summary.json'
        target.write_text(json.dumps(summary, indent=2) + '\n')
        print(f'\nALL SELECTED MODELS PASSED. Summary: {target}', flush=True)
    else:
        print('\nDRY RUN ONLY (no dependency/data checks); add --execute on A100 to perform real training.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
