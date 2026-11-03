#!/usr/bin/env python3
"""Optional graph-operator sanity tests on tiny structures.

Does not establish the 128-neighbor cap was inactive in original large clusters.
"""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models'/'dimenetpp'))
from pyg_native_graph_ops import native_radius_graph, native_triplets


class TestNativeGraphOps(unittest.TestCase):
    def test_radius_trivial(self):
        pos=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[5.,5.,5.]])
        batch=torch.zeros(4,dtype=torch.long)
        edges=native_radius_graph(pos,r=1.5,batch=batch,max_num_neighbors=128)
        actual=set(map(tuple,edges.T.tolist()))
        expected={(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)}
        self.assertEqual(actual,expected)

    def test_triplet_unordered_reference(self):
        pos=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        edges=native_radius_graph(pos,r=1.5,batch=torch.zeros(3,dtype=torch.long),max_num_neighbors=128)
        _,_,i,j,k,idx_kj,idx_ji=native_triplets(edges,3)
        got=sorted(zip(k.tolist(),j.tolist(),i.tolist()))
        s,d=edges
        naive=sorted((int(s[a]),int(d[a]),int(d[b]))
                     for a in range(s.numel()) for b in range(s.numel())
                     if int(d[a])==int(s[b]) and int(s[a])!=int(d[b]))
        self.assertEqual(got,naive)
        self.assertEqual(len(idx_ji),len(idx_kj))

    @unittest.skipUnless(importlib.util.find_spec('torch_cluster') is not None,'torch_cluster unavailable')
    def test_radius_vs_reference_when_neighbor_cap_is_inactive(self):
        try:
            from torch_cluster import radius_graph
        except (ImportError, OSError) as e:
            self.skipTest(f"Optional torch_cluster binary cannot be loaded: {e}")
        pos=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[5.,5.,5.]],dtype=torch.float)
        batch=torch.zeros(4,dtype=torch.long)
        a=native_radius_graph(pos,r=1.5,batch=batch,max_num_neighbors=128)
        b=radius_graph(pos,r=1.5,batch=batch,loop=False,max_num_neighbors=128,flow='source_to_target')
        self.assertEqual(set(map(tuple,a.T.tolist())),set(map(tuple,b.T.tolist())))
if __name__=='__main__': unittest.main()
