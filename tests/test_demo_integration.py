"""No-GPU tests of the bundled synthetic, portable, six-model demo setup."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import pandas as pd
import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import run_demo
from run_matched import RUNNERS


class TestDemoBundle(unittest.TestCase):
    def test_portable_synthetic_data(self):
        info = run_demo.preflight(ROOT / 'data/demo', 'initial', list(RUNNERS))
        self.assertEqual(info['demo_rows'], 32)
        self.assertEqual(info['n_systems'], 16)
        self.assertTrue(info['input_paths_portable'])

    def test_314_schema_matches_author_header(self):
        schema = json.loads((ROOT / 'data/descriptor_columns.json').read_text())
        d = pd.read_csv(ROOT / 'data/demo/descriptors.csv')
        self.assertEqual(len(schema), 314)
        self.assertEqual(schema, d.columns.tolist())
        self.assertEqual(len(d), 32)

    def test_atom_counts_targets_and_94_features(self):
        m = pd.read_csv(ROOT / 'data/demo/manifest.csv')
        d = pd.read_csv(ROOT / 'data/demo/descriptors.csv')
        schema = d.columns.tolist()
        self.assertEqual(len(schema[58:122] + schema[284:314]), 94)
        self.assertTrue(np.isfinite(d[schema[58:122] + schema[284:314]].to_numpy(dtype=float)).all())
        self.assertTrue((m.groupby('system_id').target.nunique() == 1).all())
        for r in m.itertuples(index=False):
            solute = Chem.AddHs(Chem.MolFromSmiles(r.solute_smiles)).GetNumAtoms()
            solvent = Chem.AddHs(Chem.MolFromSmiles(r.solvent_smiles)).GetNumAtoms()
            with (ROOT / r.input_xyz).open() as f:
                actual = int(f.readline().strip())
            self.assertEqual(actual, solute + solvent * r.n_solvent)

    def test_all_six_commands_use_demo_safe_params(self):
        for model in RUNNERS:
            cmd = run_demo.make_command(model, ROOT / 'data/demo/manifest.csv',
                                        ROOT / 'data/demo/descriptors.csv',
                                        ROOT / 'run_outputs/demo' / model,
                                        'initial', 0, 1)
            self.assertIn('--n-folds', cmd)
            self.assertEqual(cmd[cmd.index('--n-folds') + 1], '2')
            self.assertEqual(cmd[cmd.index('--epochs') + 1], '1')
            self.assertEqual(cmd[cmd.index('--cutoff') + 1], '4.0')
            if model.endswith('Dim'):
                self.assertEqual(cmd[cmd.index('--reference-folds') + 1], 'none')
            if model[0] in 'EZ':
                self.assertIn('--descriptors', cmd)

    def test_demo_dryrun(self):
        cmd = [sys.executable, str(SCRIPTS / 'run_demo.py'), '--models'] + list(RUNNERS)
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.count(' --n-folds 2 '), 6)
        self.assertIn('DRY RUN ONLY', p.stdout)

if __name__ == '__main__':
    unittest.main()
