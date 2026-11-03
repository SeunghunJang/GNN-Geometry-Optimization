#!/usr/bin/env python3
"""Launch exactly specified 5 A matched Intersection-set models.

Dry-run is the default. This launcher is newly authored and does not alter the
preserved original training source files. All prerequisites must be supplied.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

STAGES = ('initial','loose_t10','loose_t20','loose_t30',
          'tight_t10','tight_t20','tight_t30','final')
RUNNERS = {
    'B3D': 'models/3dmpnn/run_3dmpnn_B_original.py',
    'E3D': 'experiment_scripts/run_3dmpnn_E_original.py',
    'Z3D': 'experiment_scripts/run_3dmpnn_Z.py',
    'BDim': 'models/dimenetpp/run_organic_geometry_PyG_B.py',
    'EDim': 'models/dimenetpp/run_organic_geometry_PyG_E.py',
    'ZDim': 'models/dimenetpp/run_organic_geometry_PyG_Z.py',
}


def preflight(manifest: Path, desc: Path, folds: Path, models: list[str], stages: list[str]):
    import pandas as pd
    need_desc = any(x in ('E3D','Z3D','EDim','ZDim') for x in models)
    need_folds = any(x.endswith('Dim') for x in models)
    if not manifest.is_file():
        raise SystemExit(f'Missing manifest: {manifest}')
    df = pd.read_csv(manifest, low_memory=False)
    required = {'system_id','stage_mode','solute_smiles','solvent_smiles','solvent','n_solvent','input_xyz','target'}
    if not required.issubset(df.columns):
        raise SystemExit(f'Manifest missing columns: {sorted(required - set(df.columns))}')
    if df.duplicated(['system_id','stage_mode']).any():
        raise SystemExit('Manifest contains duplicated (system_id,stage_mode) rows')
    if len(df) != 3480 or set(df.stage_mode.astype(str)) != set(STAGES):
        raise SystemExit('Expected the 435 x 8 = 3480 row matched manifest, not Available-set')
    for stage in STAGES:
        chunk=df[df.stage_mode==stage]
        if len(chunk)!=435 or chunk.system_id.astype(str).nunique()!=435:
            raise SystemExit(f'Unexpected matched count in {stage}: {len(chunk)} rows')
    if any(set(df[df.stage_mode == st].system_id.astype(str)) != set(df[df.stage_mode == 'initial'].system_id.astype(str)) for st in STAGES):
        raise SystemExit('Stage membership differs across the eight matched stages')
    if need_desc:
        if not desc.is_file():
            raise SystemExit(f'Original descriptor table missing: {desc}')
        head=pd.read_csv(desc, nrows=0)
        if len(head.columns)!=314:
            raise SystemExit(f'Expected exactly 314 descriptor columns, found {len(head.columns)}')
        checks={58:'rg_total',121:'contact_5p0_solvent_heavy_atom_fraction',
                284:'solvent_molecule_min_dist_to_solute_min',313:'solvent_molecule_within_5p0_fraction'}
        for idx,name in checks.items():
            if head.columns[idx]!=name:
                raise SystemExit(f'Descriptor schema mismatch at column {idx}: {head.columns[idx]!r}')
    if need_folds:
        if not folds.is_file():
            raise SystemExit(f'Exact reference outer folds missing: {folds}')
        foldcols = set(pd.read_csv(folds,nrows=0).columns)
        required_fold_cols={'system_id','split','seed','fold','heldout_solvent'}
        if not required_fold_cols.issubset(foldcols):
            raise SystemExit(f'Reference folds missing columns: {sorted(required_fold_cols - foldcols)}')
    # The publication data archive may require remapping legacy input_xyz paths.
    selected = df[df.stage_mode.isin(stages)]
    paths = [Path(s) for s in selected.input_xyz.astype(str).unique()]
    relative = [str(p) for p in paths if not p.is_absolute()]
    if relative:
        raise SystemExit(f'{len(relative)} input_xyz values are relative, e.g. {relative[0]!r}. Remap to absolute XYZ paths before executing.')
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} referenced XYZ files missing on disk, e.g. {missing[0]!r}. Remap the manifest before rerunning.')
    print(f'PREFLIGHT PASS: {len(df)} matched rows; descriptors={need_desc}; reference folds={need_folds}.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--descriptors',type=Path,default=Path('MISSING_ORIGINAL_DESCRIPTORS.csv'))
    p.add_argument('--reference-folds',type=Path,default=Path('MISSING_REFERENCE_OUTER_FOLDS.csv'))
    p.add_argument('--outroot',type=Path,default=Path('run_outputs'))
    p.add_argument('--models',nargs='+',choices=list(RUNNERS),default=list(RUNNERS))
    p.add_argument('--stages',default=','.join(STAGES))
    p.add_argument('--execute',action='store_true',help='Explicitly opt in to training; dry-run otherwise')
    p.add_argument('--dry-run',action='store_true',help='Print commands only (default)')
    args=p.parse_args()
    if args.dry_run and args.execute: p.error('--dry-run and --execute cannot be combined')
    stages=[s.strip() for s in args.stages.split(',') if s.strip()]
    if not stages or len(stages)!=len(set(stages)) or not set(stages).issubset(STAGES):
        p.error(f'--stages must be a unique subset of {STAGES}')
    root=Path(__file__).resolve().parents[1]
    outroot=args.outroot.expanduser().resolve()
    manifest=args.manifest.expanduser().resolve()
    desc=args.descriptors.expanduser().resolve()
    folds=args.reference_folds.expanduser().resolve()
    if args.execute:
        preflight(manifest,desc,folds,args.models,stages)
    for model in args.models:
        runner=root/RUNNERS[model]
        cmd=[sys.executable,'-u',str(runner),
             '--manifest',str(manifest),
             '--outdir',str(outroot/model),
             '--stages',','.join(stages),
             '--splits','random,loso','--seeds','0,1,2',
             '--n-folds','5','--epochs','600','--patience','100',
             '--val-frac','0.15','--batch-size','16',
             '--hidden-dim','96','--num-layers','4',
             '--n-rbf','32','--cutoff','5.0','--dropout','0.10',
             '--lr','0.001','--weight-decay','0.00001']
        if model in ('E3D','Z3D','EDim','ZDim'):
            cmd.extend(['--descriptors',str(desc),'--desc-hidden','32'])
        if model.endswith('Dim'):
            cmd.extend(['--architecture','dimenetpp','--max-num-neighbors','128',
                        '--role-offset','100','--dimenet-radial','6','--dimenet-spherical','7',
                        '--dimenet-int-emb','64','--dimenet-basis-emb','8','--dimenet-out-emb','128',
                        '--reference-folds',str(folds)])
        print(f'\n[{model}] '+shlex.join(cmd),flush=True)
        if not args.execute: continue
        (outroot/model).mkdir(parents=True,exist_ok=True)
        subprocess.run(cmd,check=True,cwd=root,env=os.environ.copy())
    if not args.execute:
        print('\nDRY RUN ONLY; add --execute after acquiring the exact original manifest, XYZ, descriptors and fold table.',flush=True)

if __name__=='__main__': main()
