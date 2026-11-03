"""CUDA-extension-free spatial graph and angle-triplet operations for PyG 2.6.1.

This substitutes ONLY PyG's neighborhood construction (torch-cluster) and
triplet indexing (torch-sparse), not SchNet/DimeNet++ trainable layers.
Designed for glibc 2.31 / PyTorch 2.10 + cu128 where official extension
binaries can fail to load. Operations run on the same GPU as positions.

Spatial cutoff rule: Euclidean distance < r, excluding self edges; duplicate
atoms at different indices are retained if within the cutoff. The native
implementation chooses nearest neighbors if the per-atom cap is exceeded,
which might differ from torch-cluster selection. Use max_num_neighbors larger
than all observed neighbor counts for matched connectivity.
"""
from __future__ import annotations

from typing import Optional, Tuple
import torch
from torch import Tensor, nn


def native_radius_graph(
    pos: Tensor,
    r: float,
    batch: Optional[Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    """Return directed source->target edges for each graph using torch.cdist.

    No cross-graph edges; max neighbor count enforced per *target* atom.
    All selection is discrete and does not require coordinate gradients.
    Distances in model layers themselves remain differentiable.
    """
    del num_workers, batch_size
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError('pos must be [num_atoms,3]')
    if flow not in ('source_to_target', 'target_to_source'):
        raise ValueError('unsupported flow: ' + str(flow))
    if r <= 0 or max_num_neighbors <= 0:
        raise ValueError('r and max_num_neighbors must be positive')
    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    if batch.shape[0] != pos.shape[0] or batch.device != pos.device:
        raise ValueError('batch must correspond to pos on same device')

    edges = []
    with torch.no_grad():
        for gid in torch.unique(batch, sorted=True):
            atom_ids = (batch == gid).nonzero(as_tuple=True)[0]
            n = int(atom_ids.numel())
            if n == 0 or (n == 1 and not loop):
                continue
            distances = torch.cdist(pos[atom_ids].detach(), pos[atom_ids].detach())
            within = distances < float(r)
            if not loop:
                within.fill_diagonal_(False)
            # A complete directed adjacency is cheap for the existing <=386 atom
            # clusters and avoids uncertainty in GPU-native radius implementations.
            if n <= max_num_neighbors + int(not loop):
                target_local, source_local = within.nonzero(as_tuple=True)
            else:
                k = min(max_num_neighbors, n if loop else n - 1)
                masked = distances.masked_fill(~within, float('inf'))
                d, candidate = torch.topk(masked, k=k, largest=False,
                                          sorted=True, dim=1)
                keep = torch.isfinite(d)
                target_local = torch.arange(n, device=pos.device).view(n, 1).expand_as(candidate)[keep]
                source_local = candidate[keep]
            if target_local.numel():
                source = atom_ids[source_local]
                target = atom_ids[target_local]
                if flow == 'target_to_source':
                    source, target = target, source
                edges.append(torch.stack((source, target), dim=0))
    if edges:
        return torch.cat(edges, dim=1)
    return torch.empty((2, 0), dtype=torch.long, device=pos.device)


def native_triplets(edge_index: Tensor, num_nodes: int) -> Tuple[Tensor, ...]:
    """Replicate PyG DimeNet k->j->i triplet indexing without torch-sparse.

    Returns (i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji). Incoming
    edges to atom j are enumerated for each edge j->i, excluding k==i.
    Uses vectorized per-atom edge lists; no Python loops over edges.
    """
    if edge_index.shape[0] != 2:
        raise ValueError('edge_index must be [2,E]')
    src, dst = edge_index.long()
    num_edges = int(src.numel())
    if num_edges == 0:
        empty = src.new_empty((0,))
        return dst, src, empty, empty, empty, empty, empty

    with torch.no_grad():
        counts = torch.bincount(dst, minlength=num_nodes)
        degree = int(counts.max().item())
        # Sort by destination atom. Positions within each row are deterministic.
        order = torch.argsort(dst, stable=True)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        offsets = torch.arange(num_edges, device=src.device) - ptr[dst[order]]
        incoming = src.new_full((num_nodes, degree), -1)
        incoming[dst[order], offsets] = order

        # For each j->i edge, collect all k->j edges.
        candidate_kj = incoming[src]                  # [E, max_degree]
        present = candidate_kj >= 0
        safe_candidate = candidate_kj.clamp_min(0)
        # Filter out immediate backtrack i->j->i.
        present = present & (src[safe_candidate] != dst[:, None])
        idx_kj = candidate_kj[present]
        idx_ji = torch.arange(num_edges, device=src.device)[:, None].expand_as(candidate_kj)[present]
        idx_i = dst[idx_ji]
        idx_j = src[idx_ji]
        idx_k = src[idx_kj]
    return dst, src, idx_i, idx_j, idx_k, idx_kj, idx_ji


class NativeTorchInteractionGraph(nn.Module):
    """Drop-in SchNet.interaction_graph accepting (pos,batch)."""
    def __init__(self, cutoff: float, max_num_neighbors: int):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor):
        edge_index = native_radius_graph(pos, self.cutoff, batch,
                                         max_num_neighbors=self.max_num_neighbors)
        src, dst = edge_index
        # Preserve normal differentiability of distance features.
        edge_weight = (pos[src] - pos[dst]).norm(dim=-1)
        return edge_index, edge_weight


def install_dimenet_extensionless_ops() -> None:
    """Patch PyG 2.6.x DimeNet module graph/triplet helpers only.

    No PyG source files or neural-network parameter modules are modified.
    Expected to be called once before the first DimeNet++ forward pass.
    """
    import importlib
    dm = importlib.import_module('torch_geometric.nn.models.dimenet')
    if not getattr(dm, '_extensionless_ops_installed', False):
        dm.radius_graph = native_radius_graph
        dm.triplets = native_triplets
        dm._extensionless_ops_installed = True
