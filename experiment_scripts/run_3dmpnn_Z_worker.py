#!/usr/bin/env python3
"""Run a fixed list of 3D-MPNN Z geometry stages sequentially on one GPU.

This runner does not change the author-supplied original Model Z implementation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

STAGES = ('initial','loose_t10','loose_t20','loose_t30','tight_t10','tight_t20','tight_t30','final')
PRED_FILE = 'organic_geometry_GNN_intersection_predictions.csv'
RESULT_FILE = 'organic_geometry_GNN_intersection_results.csv'
CONFIG_FILE = 'organic_geometry_GNN_intersection_config.json'


def verify_stage(outdir, stage, args=None):
    pred_path, result_path, config_path = [outdir / f for f in (PRED_FILE, RESULT_FILE, CONFIG_FILE)]
    if not all(p.is_file() for p in (pred_path, result_path, config_path)):
        return False, 'missing one or more CSV/config outputs'
    try:
        p = pd.read_csv(pred_path)
        r = pd.read_csv(result_path)
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
        if args is not None:
            expect = {'epochs': args.epochs,'patience': args.patience,
                      'batch_size': args.batch_size,'cutoff': args.cutoff,
                      'stages': stage, 'splits': 'random,loso','seeds': [0,1,2]}
            for k, v in expect.items():
                if cfg.get(k) != v:
                    return False, f'config mismatch: {k}={cfg.get(k)!r} expected {v!r}'
        if len(p) != 2610 or len(r) != 36:
            return False, f'row count mismatch: pred={len(p)}, results={len(r)}'
        if set(p.stage_mode) != {stage} or set(r.stage_mode) != {stage}:
            return False, 'wrong stage in CSVs'
        g = p.groupby(['split','seed'], dropna=False).size()
        expected_idx = pd.MultiIndex.from_product([['loso','random'],[0,1,2]],names=['split','seed'])
        if not g.index.equals(expected_idx) or not (g == 435).all():
            return False, f'wrong split/seed group counts: {g.to_dict()}'
        g2 = r.groupby(['split','seed'], dropna=False).size()
        if not (g2 == 7).loc[[('loso', s) for s in range(3)]].all():
            return False, 'incorrect LOSO result count'
        if not (g2 == 5).loc[[('random', s) for s in range(3)]].all():
            return False, 'incorrect Random result count'
        if p.duplicated(['split','seed','system_id']).any():
            return False, 'duplicate predictions'
        if not np.isfinite(p[['y_true','y_pred']].to_numpy(dtype=float)).all():
            return False, 'NaN or infinite predictions'
        return True, 'PASS'
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--gpu', required=True)
    pa.add_argument('--stages', required=True, help='Comma-separated stages')
    pa.add_argument('--manifest', required=True)
    pa.add_argument('--descriptors', required=True)
    pa.add_argument('--outroot', required=True)
    pa.add_argument('--project-dir', required=True)
    pa.add_argument('--epochs', type=int, default=600)
    pa.add_argument('--patience', type=int, default=100)
    pa.add_argument('--batch-size', type=int, default=16)
    pa.add_argument('--cutoff', type=float, default=5.0)
    args=pa.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    outroot=Path(args.outroot).expanduser().resolve()
    model_script=Path(__file__).resolve().parent/'run_3dmpnn_Z.py'
    stage_names = [s.strip() for s in args.stages.split(',') if s.strip()]
    if len(stage_names) != len(set(stage_names)) or not all(s in STAGES for s in stage_names):
        raise SystemExit(f'Invalid or repeated stages: {stage_names}')
    if not model_script.is_file():
        raise SystemExit(f'Missing model script: {model_script}')

    for stage in stage_names:
        sd=outroot/stage
        sd.mkdir(parents=True, exist_ok=True)
        complete_flag=sd/'complete.ok'
        status, reason=verify_stage(sd, stage, args)
        if status:
            complete_flag.write_text('Verified complete\n',encoding='utf-8')
            print(f'[SKIP VERIFIED] GPU={args.gpu} {stage}',flush=True)
            continue
        print(f'[START] GPU={args.gpu} {stage} ({reason})',flush=True)
        if complete_flag.exists():
            complete_flag.unlink()
        cmd=[sys.executable,'-u',str(model_script),
             '--manifest',str(Path(args.manifest).resolve()),
             '--descriptors',str(Path(args.descriptors).resolve()),
             '--outdir',str(sd),
             '--stages',stage,'--splits','random,loso','--seeds','0,1,2',
             '--epochs',str(args.epochs),'--patience',str(args.patience),
             '--batch-size',str(args.batch_size), '--cutoff',str(args.cutoff),
             '--n-folds','5','--val-frac','0.15','--hidden-dim','96',
             '--num-layers','4','--n-rbf','32','--desc-hidden','32',
             '--dropout','0.10','--lr','0.001','--weight-decay','0.00001']
        env=os.environ.copy()
        env['CUDA_VISIBLE_DEVICES']=args.gpu
        env['OMP_NUM_THREADS']='4'
        with (sd/'run.log').open('w',encoding='utf-8') as log:
            log.write('Command: '+ ' '.join(cmd)+'\n')
            log.flush()
            result=subprocess.run(cmd,cwd=project_dir,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        if result.returncode != 0:
            print(f'[FAILED] GPU={args.gpu} {stage}, exit_code={result.returncode}; see {sd/"run.log"}',flush=True)
            raise SystemExit(result.returncode)
        ok, reason=verify_stage(sd,stage,args)
        if not ok:
            print(f'[FAILED VALIDATION] GPU={args.gpu} {stage}: {reason}',flush=True)
            raise SystemExit(2)
        complete_flag.write_text('Verified complete '+ datetime.now(timezone.utc).isoformat()+'\n', encoding='utf-8')
        print(f'[COMPLETE] GPU={args.gpu} {stage}: 2610 predictions, 36 fold results',flush=True)
    print(f'[WORKER FINISHED] GPU={args.gpu}: {len(stage_names)} stage(s)',flush=True)

if __name__=='__main__':
    main()
