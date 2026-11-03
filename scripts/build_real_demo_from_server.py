#!/usr/bin/env python3
"""Create a portable SMALL real-data demo from the author's A100 originals.

No data is published or pushed. Default data/demo_real/ is Git-ignored.
Redistribution of original labels and coordinates requires separate approval.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from run_matched import STAGES

REPO = Path(__file__).resolve().parents[1]
REQUIRED = ['system_id', 'stage_mode', 'input_xyz', 'target', 'solute_smiles',
            'solvent_smiles', 'solvent', 'n_solvent']


def pick_ids(df: pd.DataFrame, n: int, stages: list[str]):
    groups = {}
    for solvent, g in df[df.stage_mode == 'initial'].groupby('solvent', sort=True):
        ids = g.system_id.astype(str).tolist()
        valid = []
        for sid in ids:
            present = set(df.loc[df.system_id.astype(str) == sid, 'stage_mode'].astype(str))
            if set(stages).issubset(present):
                valid.append(sid)
        groups[solvent] = valid
    selected = []
    for round_no in range(max(map(len, groups.values()))):
        for solvent in sorted(groups):
            if len(selected) == n:
                return selected
            if round_no < len(groups[solvent]):
                sid = groups[solvent][round_no]
                if sid not in selected:
                    selected.append(sid)
    raise ValueError(f'Could only select {len(selected)} complete systems; requested {n}')


def abs_original_path(value: str, manifest_file: Path):
    p = Path(value)
    if p.is_absolute():
        return p
    # Source manifest may itself use relative paths from its own parent folder.
    if (manifest_file.parent / p).exists():
        return manifest_file.parent / p
    if (REPO / p).exists():
        return REPO / p
    return manifest_file.parent / p


def check_xyz(path: Path, n_expected: int):
    if not path.is_file():
        raise ValueError(f'Missing original XYZ: {path}')
    with path.open() as f:
        n = int(f.readline().strip())
    if n != n_expected:
        raise ValueError(f'Atom-count mismatch in source XYZ {path}: {n} != {n_expected}')


def mol_natoms(smiles: str) -> int:
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        raise ValueError(f'Invalid SMILES in source: {smiles}')
    return Chem.AddHs(m).GetNumAtoms()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True, type=Path,
                    help='Original 3,480-row intersection manifest on the A100 server')
    ap.add_argument('--descriptors', required=True, type=Path,
                    help='Original 3,480-row, 314-column descriptor CSV')
    ap.add_argument('--output', default='data/demo_real', type=Path,
                    help='Repository-relative folder; by default ignored by Git')
    ap.add_argument('--systems', default=14, type=int)
    ap.add_argument('--stages', default='initial,final',
                    help='Comma-separated subset of eight stage names, or all')
    ap.add_argument('--replace-synthetic', action='store_true',
                    help='Only needed to replace the bundled synthetic fixture with an approved real demo')
    ap.add_argument('--confirm-public-release', action='store_true',
                    help='Confirm distribution rights when writing real data to data/demo/')
    args = ap.parse_args()
    if args.systems < 8:
        ap.error('--systems must be >=8 to support 2-fold training and validation')
    stages = list(STAGES) if args.stages == 'all' else [s.strip() for s in args.stages.split(',') if s.strip()]
    if not stages or len(set(stages)) != len(stages) or not set(stages).issubset(STAGES):
        ap.error(f'Invalid stages: {stages}')
    manifest_file = args.manifest.expanduser().resolve()
    desc_file = args.descriptors.expanduser().resolve()
    output = (REPO / args.output).resolve()
    if not output.is_relative_to(REPO):
        ap.error('--output must be within repository')
    if output == REPO / 'data/demo':
        if not (args.confirm_public_release and args.replace_synthetic):
            ap.error('To replace the public fixture, provide BOTH --replace-synthetic and '
                     '--confirm-public-release after data rights are approved')
    elif args.confirm_public_release or args.replace_synthetic:
        ap.error('Publication/replace flags are only for --output data/demo')
    if not manifest_file.is_file() or not desc_file.is_file():
        ap.error('Both source CSV files must exist')
    if output.exists() and output != REPO / 'data/demo':
        ap.error(f'Output folder exists: {output}. Choose a new directory to avoid data loss.')
    if output == REPO / 'data/demo' and not (output / 'SYNTHETIC_DEMO_MARKER.txt').is_file():
        ap.error('Refusing to replace a non-synthetic public demo directory')
    df = pd.read_csv(manifest_file, low_memory=False)
    desc = pd.read_csv(desc_file, low_memory=False)
    schema = json.loads((REPO / 'data/descriptor_columns.json').read_text())
    if not set(REQUIRED).issubset(df.columns) or len(df) != 3480:
        raise ValueError('Expected 3,480-row source manifest with all eight required fields')
    if len(desc) != 3480 or list(desc.columns) != schema or len(schema) != 314:
        raise ValueError('Expected the exact original 3,480-row, 314-column descriptor schema')
    keys = ['system_id', 'stage_mode']
    df['system_id'] = df.system_id.astype(str)
    desc['system_id'] = desc.system_id.astype(str)
    if df.duplicated(keys).any() or desc.duplicated(keys).any():
        raise ValueError('Original source keys are not unique')
    ids = pick_ids(df, args.systems, stages)
    selected = df[df.system_id.isin(ids) & df.stage_mode.isin(stages)].copy()
    rank_id = {v: i for i, v in enumerate(ids)}
    selected['_idrank'] = selected.system_id.map(rank_id)
    selected['_stagerank'] = selected.stage_mode.map({s: i for i, s in enumerate(stages)})
    selected = selected.sort_values(['_stagerank', '_idrank']).drop(columns=['_idrank','_stagerank'])
    if len(selected) != args.systems * len(stages):
        raise ValueError('Missing source rows among selected systems/stages')
    selected_desc = desc.set_index(keys).loc[pd.MultiIndex.from_frame(selected[keys])].reset_index()
    if not np.allclose(selected.target.to_numpy(dtype=float),
                       selected_desc.target.to_numpy(dtype=float), atol=1e-6, rtol=1e-6):
        raise ValueError('Source label mismatch between manifest and descriptors')
    if not (selected.input_xyz.astype(str).to_numpy() ==
            selected_desc.input_xyz.astype(str).to_numpy()).all():
        raise ValueError('Original XYZ paths disagree between source tables')
    if not np.isfinite(selected_desc[schema[58:122] + schema[284:314]].to_numpy(dtype=float)).all():
        raise ValueError('Source contains non-finite selected descriptors')
    if selected.groupby('system_id').target.nunique().max() != 1:
        raise ValueError('Source target changes between stages')

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.demo_build_', dir=output.parent))
    try:
        (temp / 'xyz').mkdir()
        records = selected.to_dict('records')
        copies = []
        hashes = {}
        for i, row in enumerate(records):
            old = abs_original_path(str(row['input_xyz']), manifest_file)
            atom_count = mol_natoms(row['solute_smiles']) + (
                int(row['n_solvent']) * mol_natoms(row['solvent_smiles']))
            check_xyz(old, atom_count)
            filename = f"example{rank_id[str(row['system_id'])] + 1:03d}_{row['stage_mode']}.xyz"
            target = temp / 'xyz' / filename
            shutil.copyfile(old, target)
            relative = (args.output / 'xyz' / filename).as_posix()
            row['input_xyz'] = relative
            copies.append(relative)
            sha = hashlib.sha256(target.read_bytes()).hexdigest()
            hashes[relative] = sha
        selected_out = pd.DataFrame(records)[REQUIRED]
        selected_desc['input_xyz'] = copies
        selected_out.to_csv(temp / 'manifest.csv', index=False)
        selected_desc[schema].to_csv(temp / 'descriptors.csv', index=False)
        (temp / 'FILE_SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n')
        (temp / 'README.md').write_text(
            '# Selected real-data demonstration subset\n\n'
            f'Rows: {len(selected)}; systems: {args.systems}; stages: {", ".join(stages)}.\n\n'
            'Selected from the author source tables without changing targets or '
            'descriptor values. Only XYZ paths were rewritten relative to repository root.\n\n'
            'A small diagnostic run is NOT the full 435-system article benchmark, '
            'and 2-fold demo scores must never be compared to the published tables.\n\n'
            'Confirm original-label, structure-data and institutional redistribution '
            'rights before committing or publishing this folder.\n')
        if output == REPO / 'data/demo':
            old_demo = output.parent / '.synthetic_demo_backup'
            if old_demo.exists():
                raise RuntimeError('Backup folder already exists; do not overwrite it')
            output.rename(old_demo)
            try:
                temp.rename(output)
            except BaseException:
                old_demo.rename(output)
                raise
            shutil.rmtree(old_demo)
        else:
            temp.rename(output)
        print(f'CREATED PORTABLE REAL DEMO: {len(selected)} rows; {len(ids)} systems; '
              f'{len(stages)} stages; {len(copies)} copied XYZ; folder: {output}')
        print('Paths are repo-relative; source server paths are not written to public CSVs.')
        if output != REPO / 'data/demo':
            print('PRIVATE BY DEFAULT: data/demo_real/ is Git-ignored; no public release performed.')
    finally:
        if temp.exists():
            shutil.rmtree(temp)

if __name__ == '__main__':
    main()
