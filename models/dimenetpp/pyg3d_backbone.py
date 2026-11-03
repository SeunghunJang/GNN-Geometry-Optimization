"""PyG-native SchNet/DimeNet++ backbones for the legacy B/E/Z CV runners.

The original GeometryGNN was a custom distance-gated MPNN. This module
replaces its representation network with stock PyG SchNet or DimeNetPlusPlus.
The original regression/prediction split pipelines are retained separately.
Both encoders are trained from scratch (no QM9 pretrained weights).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.models import SchNet, DimeNetPlusPlus
from pyg_native_graph_ops import NativeTorchInteractionGraph, install_dimenet_extensionless_ops
from pyg_role_embedding import AdditiveAtomRoleEmbedding

# CUDA extension independent graph construction; all learned layers remain PyG native.
# Use the same policy for the complete 3D model comparison.


class PyGGraphRegressor(nn.Module):
    """A stock PyG 3D backbone with graph-level regressor and optional descriptors.

    Differences from vanilla implementations:
      - SchNet's final scalar projection is replaced by Identity to expose the
        stock SchNet's pooled learned 48-D graph representation.
      - DimeNet++ uses its stock output blocks to produce learned graph features;
        its atom-wise sum readout is normalized to a per-atom mean.
      - For B/E/Z parity with the original experiment, atomic element and
        solute/solvent role have separate, SHARED learnable embeddings:
        embedding(z, role) = z_embedding(z) + role_embedding(role).
        A packed integer z + 100*role only passes categories through the
        stock PyG backbone interface; it is not a physical atomic number.
        This prevents untrained solvent-specific element IDs in LOSO.
      - E and Z append a trainable 94-descriptor MLP after graph representation.
    """

    def __init__(
        self, architecture: str, variant: str, hidden_dim: int = 96,
        num_layers: int = 4, n_rbf: int = 32, cutoff: float = 5.0,
        dropout: float = 0.10, desc_hidden: int = 32,
        n_descriptors: int = 94, max_num_neighbors: int = 128,
        role_offset: int = 100, dimenet_radial: int = 6,
        dimenet_spherical: int = 7, dimenet_int_emb: int = 64,
        dimenet_basis_emb: int = 8, dimenet_out_emb: int = 128,
    ) -> None:
        super().__init__()
        if architecture not in ("schnet", "dimenetpp"):
            raise ValueError(f"Unknown architecture: {architecture}")
        if variant not in ("B", "E", "Z"):
            raise ValueError(f"Unknown variant: {variant}")
        if hidden_dim < 8 or hidden_dim % 2:
            raise ValueError("hidden_dim must be even and >= 8")
        if role_offset != 100:
            raise ValueError("role_offset must be 100 for separate atom/role embeddings")
        if max_num_neighbors < 1:
            raise ValueError("max_num_neighbors must be positive")
        self.architecture = architecture
        self.variant = variant
        self.role_offset = role_offset
        self.n_descriptors = n_descriptors
        self.dropout = nn.Dropout(dropout)

        # Graph feature size is identical for both stock backbones.
        graph_dim = hidden_dim // 2
        if architecture == "schnet":
            encoder = SchNet(
                hidden_channels=hidden_dim,
                num_filters=hidden_dim,
                num_interactions=num_layers,
                num_gaussians=n_rbf,
                cutoff=cutoff,
                max_num_neighbors=max_num_neighbors,
                readout="mean",
            )
            encoder.interaction_graph = NativeTorchInteractionGraph(cutoff, max_num_neighbors)
            # Restore the original MPNN input policy: separate atom and role
            # embeddings, summed elementwise for every node.
            encoder.embedding = AdditiveAtomRoleEmbedding(hidden_dim, role_offset)
            # PyG SchNet.forward: emb -> interactions -> lin1 -> act -> lin2
            # -> graph readout. Identity makes output graph_dim, not a scalar.
            if not hasattr(encoder, "lin2"):
                raise RuntimeError("Installed PyG SchNet lacks .lin2; check version")
            encoder.lin2 = nn.Identity()
        else:
            install_dimenet_extensionless_ops()
            encoder = DimeNetPlusPlus(
                hidden_channels=hidden_dim,
                out_channels=graph_dim,
                num_blocks=num_layers,
                int_emb_size=dimenet_int_emb,
                basis_emb_size=dimenet_basis_emb,
                out_emb_channels=dimenet_out_emb,
                num_spherical=dimenet_spherical,
                num_radial=dimenet_radial,
                cutoff=cutoff,
                max_num_neighbors=max_num_neighbors,
                # Non-zero last layer helps learn graph embeddings for
                # subsequent B/E/Z fusion heads at the start of training.
                output_initializer="glorot_orthogonal",
            )
            # PyG 2.6.1 DimeNet++ uses encoder.emb.emb(z). Replacing only
            # that categorical lookup retains its radial/angle message layers.
            encoder.emb.emb = AdditiveAtomRoleEmbedding(hidden_dim, role_offset)
        self.encoder = encoder
        self.graph_dim = graph_dim

        if variant in ("E", "Z"):
            if n_descriptors <= 0:
                raise ValueError("E/Z variants require descriptor columns")
            self.desc_encoder = nn.Sequential(
                nn.Linear(n_descriptors, 64),
                nn.SiLU(),
                nn.LayerNorm(64),
                nn.Dropout(dropout),
                nn.Linear(64, desc_hidden),
                nn.SiLU(),
            )
        else:
            self.desc_encoder = None
            desc_hidden = 0

        self.head = nn.Sequential(
            nn.Linear(graph_dim + desc_hidden, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        # Source data.z contains true atom Z; role=0(solute),1(solvent).
        # The packed category exists only for PyG's fixed (z,pos,batch) API.
        z_true = data.z.long()
        role = data.role.long()
        if z_true.numel() == 0 or bool(((z_true < 1) | (z_true >= 100)).any().item()):
            raise ValueError("Atomic number must be in [1,99]")
        if bool(((role != 0) & (role != 1)).any().item()):
            raise ValueError("Role must be 0 (solute) or 1 (solvent)")
        packed = z_true + self.role_offset * role
        graph_embedding = self.encoder(packed, data.pos, data.batch)
        if graph_embedding.ndim == 1:
            graph_embedding = graph_embedding.view(1, -1)
        if graph_embedding.shape[1] != self.graph_dim:
            raise RuntimeError(
                f"Unexpected {self.architecture} feature size: "
                f"{graph_embedding.shape}; expected second dimension {self.graph_dim}"
            )

        # PyG DimeNet++ uses additive atom-wise graph readout by default;
        # normalize its learned graph features for cluster-size variation.
        if self.architecture == "dimenetpp":
            atom_counts = torch.bincount(
                data.batch, minlength=graph_embedding.size(0)
            ).to(graph_embedding.dtype).to(graph_embedding.device).clamp_min(1)
            graph_embedding = graph_embedding / atom_counts.view(-1, 1)

        if self.variant in ("E", "Z"):
            desc = data.desc
            if desc.ndim != 2 or desc.shape[1] != self.n_descriptors:
                raise ValueError("Expected shape [batch, n_descriptors] for data.desc")
            # The Z runner assigns zeros after training-only standardization.
            desc_embedding = self.desc_encoder(desc)
            graph_embedding = torch.cat([graph_embedding, desc_embedding], dim=-1)
        return self.head(self.dropout(F.silu(graph_embedding))).view(-1)
