#!/usr/bin/env python3
"""Tests for the extension-independent neighbor-cap audit script."""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from check_neighbor_cap import count_neighbors, load_xyz, path_mappings, resolve_xyz


class TestNeighborCapAudit(unittest.TestCase):
    def test_strict_cutoff_and_no_self_edge(self):
        pos = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                        [-1., 0., 0.], [0., -1., 0.], [5., 0., 0.]])
        maximum, capped_atoms, edges, near = count_neighbors(
            pos, cutoff=5.0, cap=3, near_tolerance=1e-5, chunk=2)
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        expected_counts = (d < 5.).sum(axis=1)
        self.assertEqual(int(expected_counts[0]), 4)  # Exact-cutoff atom is not counted by origin.
        self.assertEqual(maximum, int(expected_counts.max()))
        self.assertEqual(capped_atoms, int((expected_counts > 3).sum()))
        self.assertGreater(edges, 0)
        self.assertGreaterEqual(near, 2)  # Directed edge candidates on cutoff.

    def test_path_remapping_and_xyz_read(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xyz = root / 'test.xyz'
            xyz.write_text('2\ncomment\nH 0 0 0\nO 0 0 0.9\n', encoding='utf-8')
            mapping = path_mappings([f'/OLD={root}'])
            result = resolve_xyz('/OLD/test.xyz', root / 'manifest.csv', mapping)
            self.assertEqual(result, xyz)
            pos = load_xyz(result)
            self.assertEqual(pos.shape, (2, 3))
            self.assertEqual(count_neighbors(pos, cutoff=5., cap=128,
                                             near_tolerance=0.)[0], 1)

    def test_optional_pytorch_native_radius_graph_consistency(self):
        try:
            import torch
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models' / 'dimenetpp'))
            from pyg_native_graph_ops import native_radius_graph
        except (ImportError, OSError):
            self.skipTest('PyTorch native op dependency unavailable')
        pos = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                        [5., 0., 0.], [7., 7., 7.]])
        maximum, capped_atoms, edge_count, _ = count_neighbors(
            pos, cutoff=5., cap=128, near_tolerance=0.)
        torch_pos = torch.tensor(pos, dtype=torch.float64)
        graph = native_radius_graph(torch_pos, r=5.,
                                    batch=torch.zeros(len(pos), dtype=torch.long),
                                    max_num_neighbors=128)
        torch_counts = torch.bincount(graph[1], minlength=len(pos))
        self.assertEqual(int(torch_counts.max()), maximum)
        self.assertEqual(int(graph.size(1)), edge_count)
        self.assertEqual(capped_atoms, 0)


if __name__ == '__main__':
    unittest.main()
