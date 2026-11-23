#!/usr/bin/env python3
"""Run the bundled small data demo with unmodified B/E/Z model runners.

This is a functional smoke test, not a reproduction of article scores.
Defaults to a dry-run. Use --execute to perform a short training run.
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

REPO = Path(__file__).resolve().parents[1]
REQUIRED = ['system_id', 'stage_mode', 'input_xyz', 'target', 'solute_smiles',
            'solvent_smiles', 'solvent', 'n_solvent']
SELECTED = (58, 121, 284, 313)


def preflight(data_dir: Path, stage: str, models: list[str]):
    manifest = data_dir / 'manifest.csv'
    descriptors = data_dir / 'descriptors.csv'
    if not manifest.is_file() or not descriptors.is_file():
        raise ValueError(f'Missing manifest/descriptors in {data_dir}')
    df = pd.read_csv(manifest)
    desc = pd.read_csv(descriptors)
    if not set(REQUIRED).issubset(df.columns):
        raise ValueError(f'Demo manifest lacks required fields: {set(REQUIRED) - set(df.columns)}')
    schema = json.loads((REPO / 'data/descriptor_columns.json').read_text())
    if list(desc.columns) != schema or len(schema) != 314:
        raise ValueError('Descriptor header differs from original 314-column schema')
    keys = ['system_id', 'stage_mode']
    if df.duplicated(keys).any() or desc.duplicated(keys).any():
        raise ValueError('Duplicate system/stage key in manifest or descriptors')
    if len(df) != len(desc):
        raise ValueError('Manifest/descriptor row counts differ')
    merged = df.merge(desc[keys + ['target', 'input_xyz'] +
                           [schema[j] for j in SELECTED]],
                      on=keys, how='left', validate='one_to_one', indicator=True,
                      suffixes=('', '_desc'))
    if (merged._merge != 'both').any():
        raise ValueError('Missing corresponding descriptor record')
    if not np.allclose(merged.target, merged.target_desc, rtol=1e-7, atol=1e-7):
        raise ValueError('Target labels differ between manifest and descriptors')
    if (merged.input_xyz != merged.input_xyz_desc).any():
        raise ValueError('XYZ path differs between manifest and descriptors')
    if not np.isfinite(merged[[schema[j] for j in SELECTED]].to_numpy(float)).all():
        raise ValueError('Non-finite selected descriptor values')
    group = df[df.stage_mode == stage]
    n = len(group)
    if n < 8:
        raise ValueError(f'Demo needs >=8 systems in stage {stage}, got {n}')
    if group.system_id.astype(str).nunique() != n:
        raise ValueError('Repeated systems in requested stage')
    for value in group.input_xyz:
        p = Path(str(value))
        if p.is_absolute() or '..' in p.parts:
            raise ValueError(f'Non-portable relative XYZ path: {p}')
        resolved = (REPO / p).resolve()
        if not resolved.is_relative_to(REPO) or not resolved.is_file():
            raise ValueError(f'Missing or external XYZ: {p}')
        lines = resolved.read_text().splitlines()
        try:
            n_atoms = int(lines[0].strip())
        except (ValueError, IndexError):
            raise ValueError(f'Bad XYZ atom count in {p}')
        if len(lines) < 2 + n_atoms:
            raise ValueError(f'Incomplete XYZ in {p}')
    return dict(stages=sorted(df.stage_mode.unique()), demo_rows=len(df),
                selected_stage=stage, n_systems=n, descriptor_columns=314,
                models=models, input_paths_portable=True)


def make_command(model, manifest, descriptors, outdir, stage, seed, epochs):
    cmd = [sys.executable, '-u', str(REPO / RUNNERS[model]),
           '--manifest', str(manifest), '--outdir', str(outdir),
           '--stages', stage, '--splits', 'random', '--seeds', str(seed),
           '--n-folds', '2', '--epochs', str(epochs), '--patience', '1',
           '--val-frac', '0.20', '--batch-size', '8',
           '--hidden-dim', '96', '--num-layers', '4', '--n-rbf', '32',
           '--cutoff', '4.0', '--dropout', '0.10',
           '--lr', '0.001', '--weight-decay', '0.00001']
    if model in ('E3D', 'Z3D', 'EDim', 'ZDim'):
        cmd += ['--descriptors', str(descriptors), '--desc-hidden', '32']
    if model.endswith('Dim'):
        cmd += ['--architecture', 'dimenetpp', '--max-num-neighbors', '128',
                '--role-offset', '100', '--dimenet-radial', '6',
                '--dimenet-spherical', '7', '--dimenet-int-emb', '64',
                '--dimenet-basis-emb', '8', '--dimenet-out-emb', '128',
                '--reference-folds', 'none']
    return cmd


def check_result(outdir: Path, expected: int, stage: str, seed: int):
    file = outdir / 'organic_geometry_GNN_intersection_predictions.csv'
    if not file.is_file():
        raise RuntimeError(f'Prediction CSV not created: {file}')
    d = pd.read_csv(file)
    needed = {'system_id', 'y_true', 'y_pred', 'split', 'seed', 'fold', 'stage_mode'}
    if not needed.issubset(d.columns):
        raise RuntimeError(f'Prediction CSV lacks {sorted(needed - set(d.columns))}')
    if len(d) != expected:
        raise RuntimeError(f'Expected {expected} held-out demo predictions, got {len(d)}')
    if d.system_id.astype(str).nunique() != expected:
        raise RuntimeError('Unexpected repeated/missing held-out system')
    if set(d.split.astype(str)) != {'random'} or set(d.stage_mode.astype(str)) != {stage}:
        raise RuntimeError('Unexpected split/stage in demo predictions')
    if set(d.fold.astype(int)) != {0, 1} or set(d.seed.astype(int)) != {seed}:
        raise RuntimeError('Unexpected folds or seed in demo predictions')
    if not np.isfinite(d[['y_true', 'y_pred']].to_numpy(float)).all():
        raise RuntimeError('Non-finite demo prediction values')
    return dict(ok=True, heldout_predictions=expected, folds=[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=Path('data/demo'))
    ap.add_argument('--models', nargs='+', choices=list(RUNNERS), default=['B3D'])
    ap.add_argument('--stage', choices=STAGES, default='initial')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--outroot', type=Path, default=Path('run_outputs/demo'))
    ap.add_argument('--execute', action='store_true', help='Actually train; default prints commands only')
    ap.add_argument('--inspect', action='store_true', help='Validate the demo files without training')
    args = ap.parse_args()
    if args.epochs < 1 or args.epochs > 5:
        ap.error('Demo epochs must be between 1 and 5; use run_matched.py for paper runs')
    data_dir = (REPO / args.data_dir).resolve()
    outroot = (REPO / args.outroot).resolve()
    if not data_dir.is_relative_to(REPO):
        ap.error('Demo folder must be inside repository')
    if not outroot.is_relative_to(REPO):
        ap.error('Outputs must stay within the repository ignored folder')
    if not outroot.is_relative_to(REPO / 'run_outputs'):
        ap.error('For safety, output must be under run_outputs/ (ignored by Git)')
    try:
        info = preflight(data_dir, args.stage, args.models)
    except (ValueError, OSError, KeyError) as e:
        ap.error(str(e))
    print('DEMO PREFLIGHT PASS:', json.dumps(info, sort_keys=True), flush=True)
    if args.inspect:
        return 0
    if args.execute:
        try:
            import torch
            import torch_geometric
            import rdkit
        except (ImportError, OSError) as e:
            ap.error(f'Install PyTorch, PyG, RDKit before training: {e}')
        print(f'RUNNING DEMO ON {"GPU" if torch.cuda.is_available() else "CPU"}', flush=True)

    summary = dict(dataset='synthetic' if (data_dir / 'SYNTHETIC_DEMO_MARKER.txt').exists()
                   else 'user-provided', scientific_benchmark=False, stage=args.stage,
                   seed=args.seed, epochs=args.epochs, models={})
    for model in args.models:
        outdir = outroot / model
        cmd = make_command(model, data_dir / 'manifest.csv', data_dir / 'descriptors.csv',
                           outdir, args.stage, args.seed, args.epochs)
        print(f'[{model}] {shlex.join(cmd)}', flush=True)
        if not args.execute:
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        with (outdir / 'demo.log').open('w') as log:
            result = subprocess.run(cmd, cwd=REPO, env=os.environ.copy(),
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            lines = (outdir / 'demo.log').read_text(errors='replace').splitlines()[-45:]
            print('\n'.join(lines), file=sys.stderr)
            raise RuntimeError(f'Model {model} failed with return code {result.returncode}; see {outdir}/demo.log')
        summary['models'][model] = check_result(outdir, info['n_systems'], args.stage, args.seed)
        print(f'PASS {model}: {info["n_systems"]} held-out predictions, 2 folds', flush=True)
    if args.execute:
        outroot.mkdir(parents=True, exist_ok=True)
        (outroot / 'DEMO_PASS_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('ALL SELECTED DEMO MODELS PASSED. This is NOT a scientific benchmark.', flush=True)
    else:
        print('DRY RUN ONLY; bash scripts/run_demo.sh to perform a short model run.', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
