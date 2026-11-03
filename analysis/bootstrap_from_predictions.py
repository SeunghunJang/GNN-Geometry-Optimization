#!/usr/bin/env python3
"""Independent, source-specified solvent-group bootstrap for matched predictions.

Reads six 435-system held-out prediction files/trees; does not retrain models.
The sampled units are seven solvent groups. Paired solvent draw indices are
shared across models and geometry stages. Do not apply to Available-set CSVs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from verify_matched_predictions import (STAGES, MODELS, load_model, audit)


def rmse(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((pred-y)**2)))


def mean_absolute(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(pred-y)))


def r2(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    den = np.sum((y - y.mean())**2)
    return float(1.0 - np.sum((y - pred)**2) / den) if den>0 else float('nan')


def bootstrap_metrics(files, replicates=20000, seed=20261019, check_tables=True):
    if replicates < 100: raise ValueError('Require at least 100 draws')
    # Always validate the complete matched pool; article-table matching is optional.
    _, failures, _ = audit(files, tolerance=0.00011, check_tables=check_tables)
    if failures:
        raise ValueError(f'Supplied predictions disagree with Tables 3-4 ({len(failures)} RMSE values)')
    raw={m:load_model(m,files[m]) for m in MODELS}
    solvents=sorted(set(raw['B3D'].solvent))
    if len(solvents)!=7: raise ValueError(f'Expected seven solvents, found {solvents}')
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,7,size=(replicates,7))
    weights=np.stack([(draws==j).sum(axis=1) for j in range(7)],axis=1).astype(float)
    boot={}; metrics=[]; solvent_delta=[]
    for model in MODELS:
        df=raw[model]
        if set(df.solvent)!=set(solvents):
            raise ValueError(f'Solvent mismatch in {model}')
        for split in ('random','loso'):
            for stage in STAGES:
                a=df[(df.split==split)&(df.stage_mode==stage)]
                metric=dict(model=model,split=split,stage=stage,n=len(a),
                            rmse=rmse(a.y_true,a.y_pred),
                            mae=mean_absolute(a.y_true,a.y_pred),
                            r2=r2(a.y_true,a.y_pred))
                metrics.append(metric)
                sse=[]; count=[]
                for solvent in solvents:
                    b=a[a.solvent==solvent]
                    if len(b)==0:raise ValueError(f'Empty solvent {solvent} in {model}/{split}/{stage}')
                    sse.append(np.sum((b.y_true.to_numpy()-b.y_pred.to_numpy())**2))
                    count.append(len(b))
                num=weights@np.asarray(sse,dtype=float)
                den=weights@np.asarray(count,dtype=float)
                boot[(model,split,stage)]=np.sqrt(num/den)
        # Calculate LOSO per-solvent endpoint contrasts from 3x held-out rows.
        for solvent in solvents:
            def score(stage):
                rows=df[(df.split=='loso')&(df.stage_mode==stage)&(df.solvent==solvent)]
                return rmse(rows.y_true,rows.y_pred)
            ri=score('initial');rf=score('final')
            solvent_delta.append(dict(model=model,solvent=solvent,rmse_initial=ri,rmse_final=rf,
                                      final_minus_initial=rf-ri))
    table=pd.DataFrame(metrics)
    contrasts=[]
    def add_contrast(name,split,stage,left,right):
        a=table[(table.model==left[0])&(table.split==split)&(table.stage==left[1])].iloc[0]
        b=table[(table.model==right[0])&(table.split==split)&(table.stage==right[1])].iloc[0]
        vals=boot[(left[0],split,left[1])]-boot[(right[0],split,right[1])]
        low,high=np.percentile(vals,[2.5,97.5])
        contrasts.append(dict(contrast=name,split=split,stage=stage,
                              lhs_model=left[0],lhs_stage=left[1],
                              rhs_model=right[0],rhs_stage=right[1],
                              estimate=a.rmse-b.rmse,ci95_low=float(low),ci95_high=float(high),
                              n_bootstrap=replicates,bootstrap_seed=seed,group_count=7))
    for split in ('random','loso'):
        for model in MODELS:
            add_contrast('final_minus_initial',split,'final_vs_initial',(model,'final'),(model,'initial'))
        for prefix in ('3D','Dim'):
            for stage in STAGES:
                add_contrast('descriptor_E_minus_zero_Z',split,stage,('E'+prefix,stage),('Z'+prefix,stage))
    return table,pd.DataFrame(contrasts),pd.DataFrame(solvent_delta)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True,help='JSON mapping six models to their matched prediction CSV or stage directories')
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--replicates',type=int,default=20000)
    p.add_argument('--seed',type=int,default=20261019)
    p.add_argument('--skip-table-check',action='store_true',
                   help='Accept structurally valid new model predictions even if their pooled RMSE differs from the article')
    args=p.parse_args()
    files=json.loads(args.inputs.read_text(encoding='utf-8'))
    metrics,contrasts,solvent=bootstrap_metrics(files,args.replicates,args.seed,
                                                  check_tables=not args.skip_table_check)
    args.outdir.mkdir(parents=True,exist_ok=True)
    for name,df in [('matched_pooled_metrics.csv',metrics),
                    ('matched_bootstrap_contrasts.csv',contrasts),
                    ('matched_loso_solvent_contrasts.csv',solvent)]:
        target=args.outdir/name
        df.to_csv(target,index=False)
        print(f'WROTE {target} ({len(df)} rows)',flush=True)
    metadata={'method':'paired seven-solvent cluster bootstrap',
              'rng':'numpy.random.default_rng(seed).integers(0,7,size=(replicates,7))',
              'seed':args.seed,'replicates':args.replicates,
              'article_table_check_performed':not args.skip_table_check,
              'confidence_interval':'2.5th and 97.5th percentile; exploratory',
              'source':'independent implementation, not verified to reproduce the article CI values'}
    (args.outdir/'bootstrap_method.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print('NOTE: 7 independent solvent groups; percentile intervals are exploratory.')
    print('NOTE: this independently implemented RNG may yield different CIs from the original article. Compare to Table S5 before claiming exact replication.')
if __name__=='__main__': main()
