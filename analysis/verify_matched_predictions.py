#!/usr/bin/env python3
"""Audit 435-system matched predictions against Tables 3-4 of the article.

Input JSON maps B3D,E3D,Z3D,BDim,EDim,ZDim to one CSV or a stage-root directory.
The tool is read-only except for its JSON report. It does NOT train any model.
Never point a model input at a parent that contains both Available and Intersection results.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import pandas as pd
import numpy as np

STAGES = ['initial','loose_t10','loose_t20','loose_t30','tight_t10','tight_t20','tight_t30','final']
MODELS = ['B3D','E3D','Z3D','BDim','EDim','ZDim']
EXPECTED = {
 'random': {
 'B3D':[.8352,.9831,.8912,.8908,.8039,.8235,.8334,.8181],
 'E3D':[.9204,1.0415,1.1203,1.0388,1.0125,.9971,1.0136,1.0049],
 'Z3D':[.8011,.9945,.9324,.9129,.8242,.8364,.8304,.8544],
 'BDim':[.6976,.7527,.7355,.7169,.6811,.6876,.6811,.6772],
 'EDim':[.7750,.9116,.8596,.7886,.8658,.8044,.7872,.8014],
 'ZDim':[.7267,.7538,.7189,.7049,.6809,.6791,.6783,.6694],
 },
 'loso': {
 'B3D':[1.3015,1.5314,1.6257,1.5092,1.5919,1.6674,1.6927,1.7616],
 'E3D':[1.2497,1.3675,1.3010,1.3019,1.3596,1.4110,1.4198,1.3415],
 'Z3D':[1.3351,1.3649,1.4938,1.5790,1.6755,1.4769,1.5701,1.5468],
 'BDim':[2.2367,2.4076,2.4615,2.5212,3.2734,3.6681,3.3139,3.3126],
 'EDim':[1.2006,1.2604,1.3924,1.5071,1.5724,1.4439,1.6001,1.2737],
 'ZDim':[2.6273,2.4944,2.6886,2.3529,3.2818,2.8752,3.0058,3.3884],
 }
}
REQUIRED={'system_id','stage_mode','split','seed','fold','solvent','y_true','y_pred'}

def load_model(model, value):
    p=Path(value).expanduser()
    if p.is_file(): paths=[p]
    elif p.is_dir(): paths=sorted(p.rglob('organic_geometry_GNN_intersection_predictions.csv'))
    else: raise FileNotFoundError(p)
    if not paths: raise FileNotFoundError(f'No prediction CSVs found under {p}')
    frames=[]
    for path in paths:
        df=pd.read_csv(path,low_memory=False)
        miss=REQUIRED-set(df.columns)
        if miss: raise ValueError(f'{model}: missing {sorted(miss)} in {path}')
        df=df.copy()
        df['__source_file']=str(path)
        frames.append(df)
    out=pd.concat(frames,ignore_index=True)
    out['split']=out['split'].astype(str).str.lower()
    out['stage_mode']=out['stage_mode'].astype(str).str.lower()
    out['seed']=pd.to_numeric(out['seed'],errors='raise').astype(int)
    out['system_id']=out['system_id'].astype(str)
    out['solvent']=out['solvent'].astype(str)
    out['y_true']=pd.to_numeric(out['y_true'],errors='raise')
    out['y_pred']=pd.to_numeric(out['y_pred'],errors='raise')
    if not np.isfinite(out[['y_true','y_pred']].to_numpy(dtype=float)).all():
        raise ValueError(f'{model}: non-finite targets/predictions')
    print(f'{model}: {len(paths)} CSV file(s), {len(out)} rows')
    return out

def audit(files, tolerance, check_tables=True):
    warnings=[]; data={}
    for model in MODELS:
        if model not in files: raise ValueError(f'Missing input path for {model}')
        df=load_model(model,files[model])
        if len(df)!=20880:raise ValueError(f'{model}: expected 20,880 matched predictions, got {len(df)}. Possible Available-set mix-up.')
        if set(df.stage_mode)!=set(STAGES):raise ValueError(f'{model}: missing/unexpected stages {set(df.stage_mode)^set(STAGES)}')
        if set(df.split)!={'random','loso'}:raise ValueError(f'{model}: split values {set(df.split)}')
        if set(df.seed)!={0,1,2}:raise ValueError(f'{model}: seeds {set(df.seed)}')
        for (stage,split,seed),a in df.groupby(['stage_mode','split','seed']):
            if len(a)!=435 or a.system_id.nunique()!=435:
                raise ValueError(f'{model}: {stage}/{split}/{seed}: {len(a)} rows/{a.system_id.nunique()} unique systems, expected 435')
            if split=='random':
                if a['fold'].nunique()!=5:raise ValueError(f'{model}: random {stage}/seed {seed} not five folds')
            else:
                if 'heldout_solvent' in a:
                    if a['heldout_solvent'].astype(str).nunique()!=7:
                        raise ValueError(f'{model}: LOSO {stage}/seed {seed} not seven heldouts')
                    if not (a['heldout_solvent'].astype(str)==a['solvent']).all():
                        raise ValueError(f'{model}: LOSO heldout solvent does not match test solvent')
        k=['system_id','stage_mode','split','seed']
        if df.duplicated(k).any(): raise ValueError(f'{model}: duplicate prediction key')
        data[model]=df
    anchor=data['B3D']
    anchor_map=anchor.groupby('system_id').agg(y=('y_true','first'),solvent=('solvent','first'))
    if len(anchor_map)!=435:raise ValueError('Anchor does not contain exactly 435 systems')
    anchor_keys=anchor[['system_id','stage_mode','split','seed','fold']].rename(columns={'fold':'ref_fold'})
    if anchor_keys.duplicated(['system_id','stage_mode','split','seed']).any():raise ValueError('Duplicate anchor keys')
    result=[]; bad=[]
    for model,df in data.items():
        a=df.merge(anchor_map,on='system_id',how='left',validate='many_to_one')
        if a.y.isna().any() or not np.allclose(a.y,a.y_true,atol=1e-4,rtol=0):
            raise ValueError(f'{model}: mismatched system labels')
        if not (a.solvent_x==a.solvent_y).all():raise ValueError(f'{model}: mismatched solvent labels')
        z=df.merge(anchor_keys,on=['system_id','stage_mode','split','seed'],how='inner',validate='one_to_one')
        if len(z)!=len(df) or not (z.fold.astype(str)==z.ref_fold.astype(str)).all():
            raise ValueError(f'{model}: outer split memberships differ from B3D for same seed/stage')
        for split in ['random','loso']:
            for i,stage in enumerate(STAGES):
                chunk=df[(df.stage_mode==stage)&(df.split==split)]
                rmse=float(np.sqrt(np.mean((chunk.y_pred.to_numpy()-chunk.y_true.to_numpy())**2)))
                expected=EXPECTED[split][model][i]
                within_tolerance=abs(rmse-expected)<=tolerance
                row={'model':model,'split':split,'stage':stage,'n':len(chunk),
                     'rmse':rmse,'article_rmse':expected,'delta_from_article':rmse-expected,
                     'within_article_tolerance':within_tolerance,
                     'table_check_pass':within_tolerance if check_tables else None}
                result.append(row)
                if check_tables and not within_tolerance:bad.append(row)
    return result,bad,warnings

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,help='JSON dictionary mapping B3D,E3D,Z3D,BDim,EDim,ZDim to paths')
    p.add_argument('--report',type=Path,default=Path('matched_prediction_audit.json'))
    p.add_argument('--tolerance',type=float,default=0.00011)
    p.add_argument('--write-template',action='store_true')
    p.add_argument('--skip-table-check',action='store_true',
                   help='Check matched-set structure, targets, and folds but do not require published RMSE values (for independent reruns)')
    args=p.parse_args()
    if args.write_template:
        target=Path('matched_model_prediction_paths_TEMPLATE.json')
        target.write_text(json.dumps({m:'REPLACE_WITH_CSV_OR_STAGE_ROOT' for m in MODELS},indent=2)+'\n')
        print('WROTE',target);return
    if args.inputs is None:p.error('--inputs required unless --write-template')
    cfg=json.loads(args.inputs.read_text())
    check_tables=not args.skip_table_check
    rows,bad,warnings=audit(cfg,args.tolerance,check_tables=check_tables)
    out={'structural_checks_passed':True,
         'article_table_check_performed':check_tables,
         'all_match':(not bad) if check_tables else None,
         'model_count':len(MODELS),'expected_total_predictions':125280,
         'n_metrics':len(rows),'failures':bad,'rows':rows,'warnings':warnings}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(f'WROTE {args.report}; structural checks PASS; {len(rows)} metric values')
    if bad:
        for row in bad[:20]:print('FAIL ARTICLE TABLE',row)
        raise SystemExit(2)
    if check_tables:
        print(f'PASS: all {len(rows)} pooled RMSE values match Tables 3-4 within tolerance {args.tolerance}')
    else:
        nclose=sum(bool(row['within_article_tolerance']) for row in rows)
        print(f'ARTICLE TABLE CHECK SKIPPED: {nclose}/{len(rows)} values would match at tolerance {args.tolerance}; not a failure condition')
        print('PASS: matched 435-system identity, labels, stage membership and fold-assignment checks')
if __name__=='__main__':main()
