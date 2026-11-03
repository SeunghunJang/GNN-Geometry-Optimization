"""Unit tests for the optional A100 integration-test launcher (no training)."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from smoke_train import command, check_result


class TestRealTrainingSmokeLauncher(unittest.TestCase):
    def test_six_commands_have_a_safe_minimal_setting(self):
        root = Path(__file__).resolve().parents[1]
        manifest = Path('/fake/manifest.csv')
        desc = Path('/fake/desc.csv')
        for model in ['B3D','E3D','Z3D','BDim','EDim','ZDim']:
            with self.subTest(model=model):
                cmd = command(model, root, manifest, desc,
                              Path('/tmp/smoke_diagnostic')/model, 'initial', 1, 0)
                args = ' '.join(cmd)
                for expected in ['--stages initial','--splits random','--seeds 0',
                                 '--n-folds 2','--epochs 1','--cutoff 5.0']:
                    self.assertIn(expected, args)
                if model.endswith('Dim'):
                    self.assertIn('--reference-folds none', args)
                else:
                    self.assertNotIn('--reference-folds', args)
                if model.startswith(('E','Z')):
                    self.assertIn('--descriptors /fake/desc.csv', args)
                else:
                    self.assertNotIn('--descriptors', args)

    def test_saved_outputs_must_contain_435_held_out_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            records = [{'system_id': f'system{i:04d}', 'stage_mode': 'initial',
                        'split': 'random','seed':0,'fold':i%2,
                        'y_true':float(i%13),'y_pred':float(i%13)+.1}
                       for i in range(435)]
            df = pd.DataFrame(records)
            df.to_csv(outdir/'organic_geometry_GNN_intersection_predictions.csv',index=False)
            pd.DataFrame([{'fold':0,'rmse':0.1},{'fold':1,'rmse':0.1}]).to_csv(
                outdir/'organic_geometry_GNN_intersection_results.csv',index=False)
            (outdir/'organic_geometry_GNN_intersection_config.json').write_text(
                json.dumps({'epochs':1}))
            result=check_result(outdir,'initial',0)
            self.assertTrue(result['passed'])
            self.assertEqual(result['prediction_rows'],435)
            self.assertEqual(result['folds'],[0,1])
            df.loc[0,'y_pred']=np.nan
            df.to_csv(outdir/'organic_geometry_GNN_intersection_predictions.csv',index=False)
            with self.assertRaises(RuntimeError):
                check_result(outdir,'initial',0)


if __name__ == '__main__':
    unittest.main()
