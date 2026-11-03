"""Composable atom-type and solvent-role embedding for LOSO-safe 3D GNNs.

Packed IDs are used solely to carry the two categorical variables through the
unaltered PyG SchNet/DimeNet++ forward interface. They are not atomic numbers.

The original 3D-MPNN initialized nodes as z_embedding(z) + role_embedding(role).
With this module, a solvent element/role pair can be represented using shared
trained parameters even if that exact pair was absent from training.
"""
from __future__ import annotations
import torch
from torch import nn


class AdditiveAtomRoleEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, role_offset: int = 100):
        super().__init__()
        if role_offset != 100:
            raise ValueError('role_offset must equal 100 (atomic indices 0..99)')
        self.role_offset = role_offset
        self.z_emb = nn.Embedding(100, hidden_dim)
        self.role_emb = nn.Embedding(2, hidden_dim)

    def forward(self, packed_z_role: torch.Tensor) -> torch.Tensor:
        # All legitimate atomic numbers in this dataset are < 100.
        z = packed_z_role.long() % self.role_offset
        role = torch.div(packed_z_role.long(), self.role_offset, rounding_mode='floor')
        return self.z_emb(z) + self.role_emb(role)
