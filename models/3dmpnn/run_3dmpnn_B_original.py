import os
import json
import time
import math
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool


TARGET_COL = "target"
SYSTEM_COL = "system_id"
STAGE_COL = "stage_mode"
SOLVENT_COL = "solvent"

ATOM_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}

STAGE_ORDER = [
    "initial",
    "loose_t10",
    "loose_t20",
    "loose_t30",
    "tight_t10",
    "tight_t20",
    "tight_t30",
    "final",
]


def stable_seed(*items):
    txt = "::".join(str(x) for x in items)
    h = hashlib.md5(txt.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def calc_metrics(y, p):
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": rmse(y, p),
        "r2": float(r2_score(y, p)),
        "n": int(len(y)),
    }


def mol_num_atoms_with_h(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("RDKit failed to parse SMILES: %s" % smiles)
    mol = Chem.AddHs(mol)
    return int(mol.GetNumAtoms())


def read_xyz(path):
    path = Path(path)
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    atom_lines = lines[2:2+n]

    elems = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError("Bad XYZ line in %s: %s" % (path, line))
        elem = parts[0]
        if elem not in ATOM_Z:
            raise ValueError("Unknown element %s in %s" % (elem, path))
        xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        elems.append(elem)
        coords.append(xyz)

    return elems, np.asarray(coords, dtype=np.float32)


def make_edges(pos, cutoff, min_dist=0.7):
    # pos: torch tensor [N, 3]
    # Exclude self-edges and unrealistically short near-duplicate contacts.
    with torch.no_grad():
        dist = torch.cdist(pos, pos)
        mask = (dist < cutoff) & (dist > min_dist)
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        edge_dist = dist[edge_index[0], edge_index[1]].contiguous()
    return edge_index, edge_dist


def row_to_data(row, cutoff):
    elems, coords = read_xyz(row["input_xyz"])
    z = torch.tensor([ATOM_Z[e] for e in elems], dtype=torch.long)
    pos = torch.tensor(coords, dtype=torch.float32)

    n_solute = mol_num_atoms_with_h(row["solute_smiles"])
    n_solvent_one = mol_num_atoms_with_h(row["solvent_smiles"])
    n_solvent = int(row["n_solvent"])

    expected = n_solute + n_solvent_one * n_solvent
    if expected != len(elems):
        raise ValueError(
            "Atom count mismatch for %s stage %s: expected %d, got %d"
            % (row["system_id"], row["stage_mode"], expected, len(elems))
        )

    role = torch.ones(len(elems), dtype=torch.long)
    role[:n_solute] = 0

    edge_index, edge_dist = make_edges(pos, cutoff, min_dist=0.7)

    y = torch.tensor([float(row[TARGET_COL])], dtype=torch.float32)

    data = Data(
        z=z,
        role=role,
        pos=pos,
        edge_index=edge_index,
        edge_dist=edge_dist,
        y=y,
    )

    data.system_id = str(row["system_id"])
    data.stage_mode = str(row["stage_mode"])
    data.solvent = str(row["solvent"])
    return data


class RBFLayer(nn.Module):
    def __init__(self, n_rbf=32, cutoff=5.0, gamma=10.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.gamma = gamma

    def forward(self, d):
        d = d.view(-1, 1)
        return torch.exp(-self.gamma * (d - self.centers.view(1, -1)) ** 2)


class DistanceMessageLayer(MessagePassing):
    def __init__(self, hidden_dim, n_rbf):
        super().__init__(aggr="mean")
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.msg_lin = nn.Linear(hidden_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        upd = self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(x + upd)

    def message(self, x_j, edge_attr):
        gate = self.edge_mlp(edge_attr)
        return self.msg_lin(x_j) * gate


class GeometryGNN(nn.Module):
    def __init__(self, hidden_dim=96, num_layers=4, n_rbf=32, cutoff=5.0, dropout=0.1):
        super().__init__()
        self.cutoff = cutoff
        self.rbf = RBFLayer(n_rbf=n_rbf, cutoff=cutoff)

        self.z_emb = nn.Embedding(100, hidden_dim)
        self.role_emb = nn.Embedding(2, hidden_dim)

        self.layers = nn.ModuleList([
            DistanceMessageLayer(hidden_dim=hidden_dim, n_rbf=n_rbf)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = self.z_emb(data.z) + self.role_emb(data.role)
        edge_attr = self.rbf(data.edge_dist)

        for layer in self.layers:
            x = layer(x, data.edge_index, edge_attr)
            x = self.dropout(F.silu(x))

        g_mean = global_mean_pool(x, data.batch)
        g_max = global_max_pool(x, data.batch)
        g = torch.cat([g_mean, g_max], dim=-1)

        return self.head(g).view(-1)


def build_dataset(df, cutoff):
    data_list = []
    for _, row in df.iterrows():
        data_list.append(row_to_data(row, cutoff=cutoff))
    return data_list


def train_one_model(train_data, val_data, test_data, args, seed, device):
    set_seed(seed)

    y_train = np.asarray([float(d.y.item()) for d in train_data]).reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(y_train)

    for d in train_data:
        d.y_scaled = torch.tensor([float(scaler.transform([[d.y.item()]])[0, 0])], dtype=torch.float32)
    for d in val_data:
        d.y_scaled = torch.tensor([float(scaler.transform([[d.y.item()]])[0, 0])], dtype=torch.float32)
    for d in test_data:
        d.y_scaled = torch.tensor([float(scaler.transform([[d.y.item()]])[0, 0])], dtype=torch.float32)

    model = GeometryGNN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_rbf=args.n_rbf,
        cutoff=args.cutoff,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y_scaled.view(-1).to(device)

            opt.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.item()) * batch.num_graphs

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y = batch.y_scaled.view(-1).to(device)
                pred = model(batch)
                loss = F.mse_loss(pred, y)
                val_losses.append(float(loss.item()) * batch.num_graphs)

        val_loss = sum(val_losses) / max(1, len(val_data))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_true = []
    y_pred = []
    meta = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_scaled = model(batch).detach().cpu().numpy().reshape(-1, 1)
            pred = scaler.inverse_transform(pred_scaled).reshape(-1)

            true = batch.y.detach().cpu().numpy().reshape(-1)

            y_true.extend(true.tolist())
            y_pred.extend(pred.tolist())

            # DataLoader keeps list-like attributes
            systems = batch.system_id
            solvents = batch.solvent
            stages = batch.stage_mode

            for sid, solv, st in zip(systems, solvents, stages):
                meta.append((sid, solv, st))

    return np.asarray(y_true), np.asarray(y_pred), meta, epoch, best_val


def make_val_split(train_idx, seed, val_frac=0.15):
    rng = np.random.RandomState(seed)
    idx = np.asarray(train_idx)
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * val_frac)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return tr_idx, val_idx


def run_stage(stage_df, args, device):
    stage = str(stage_df[STAGE_COL].iloc[0])
    print("=" * 80)
    print("Building dataset for stage:", stage, "rows:", len(stage_df), flush=True)

    data_list = build_dataset(stage_df, cutoff=args.cutoff)
    y_all = stage_df[TARGET_COL].to_numpy(dtype=float)

    results = []
    preds = []

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    if "random" in splits:
        for seed in args.seeds:
            kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=seed)

            for fold, (trainval_idx, test_idx) in enumerate(kf.split(np.arange(len(data_list)))):
                split_seed = stable_seed("geometry_gnn", "random", seed, fold)
                tr_idx, val_idx = make_val_split(trainval_idx, split_seed, val_frac=args.val_frac)

                train_data = [data_list[i] for i in tr_idx]
                val_data = [data_list[i] for i in val_idx]
                test_data = [data_list[i] for i in test_idx]

                print("stage", stage, "random seed", seed, "fold", fold,
                      "train", len(train_data), "val", len(val_data), "test", len(test_data), flush=True)

                y_true, y_hat, meta, last_epoch, best_val = train_one_model(
                    train_data, val_data, test_data, args, split_seed, device
                )

                m = calc_metrics(y_true, y_hat)
                results.append({
                    "experiment": "B_geometry_only_GNN",
                    "stage_mode": stage,
                    "split": "random",
                    "seed": seed,
                    "fold": fold,
                    "heldout_solvent": "",
                    "n_train": len(train_data),
                    "n_val": len(val_data),
                    "n_test": len(test_data),
                    "last_epoch": last_epoch,
                    "best_val_loss_scaled": best_val,
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "r2": m["r2"],
                })

                for (sid, solv, st), yt, yp in zip(meta, y_true, y_hat):
                    preds.append({
                        "experiment": "B_geometry_only_GNN",
                        "stage_mode": stage,
                        "split": "random",
                        "seed": seed,
                        "fold": fold,
                        "heldout_solvent": "",
                        "system_id": sid,
                        "solvent": solv,
                        "y_true": float(yt),
                        "y_pred": float(yp),
                    })

    if "loso" in splits:
        solvents = sorted(stage_df[SOLVENT_COL].astype(str).unique())

        for seed in args.seeds:
            for heldout in solvents:
                test_idx = np.where(stage_df[SOLVENT_COL].astype(str).to_numpy() == heldout)[0]
                trainval_idx = np.where(stage_df[SOLVENT_COL].astype(str).to_numpy() != heldout)[0]

                split_seed = stable_seed("geometry_gnn", "loso", heldout, seed)
                tr_idx, val_idx = make_val_split(trainval_idx, split_seed, val_frac=args.val_frac)

                train_data = [data_list[i] for i in tr_idx]
                val_data = [data_list[i] for i in val_idx]
                test_data = [data_list[i] for i in test_idx]

                print("stage", stage, "loso seed", seed, heldout,
                      "train", len(train_data), "val", len(val_data), "test", len(test_data), flush=True)

                y_true, y_hat, meta, last_epoch, best_val = train_one_model(
                    train_data, val_data, test_data, args, split_seed, device
                )

                m = calc_metrics(y_true, y_hat)
                results.append({
                    "experiment": "B_geometry_only_GNN",
                    "stage_mode": stage,
                    "split": "loso",
                    "seed": seed,
                    "fold": -1,
                    "heldout_solvent": heldout,
                    "n_train": len(train_data),
                    "n_val": len(val_data),
                    "n_test": len(test_data),
                    "last_epoch": last_epoch,
                    "best_val_loss_scaled": best_val,
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "r2": m["r2"],
                })

                for (sid, solv, st), yt, yp in zip(meta, y_true, y_hat):
                    preds.append({
                        "experiment": "B_geometry_only_GNN",
                        "stage_mode": stage,
                        "split": "loso",
                        "seed": seed,
                        "fold": -1,
                        "heldout_solvent": heldout,
                        "system_id": sid,
                        "solvent": solv,
                        "y_true": float(yt),
                        "y_pred": float(yp),
                    })

    return results, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="organic_GNN_intersection_manifest.csv")
    parser.add_argument("--outdir", default="organic_geometry_GNN_intersection_results")
    parser.add_argument("--stages", default="initial,tight_t20,final")
    parser.add_argument("--splits", default="random,loso")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--n-rbf", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    args = parser.parse_args()

    args.seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("device:", device)
    print("manifest:", args.manifest)
    print("outdir:", outdir)
    print("args:", vars(args), flush=True)

    df = pd.read_csv(args.manifest)

    if args.stages == "all":
        stages = STAGE_ORDER
    else:
        stages = [x.strip() for x in args.stages.split(",") if x.strip()]

    all_results = []
    all_preds = []

    t0 = time.time()

    for stage in stages:
        stage_df = df[df[STAGE_COL].astype(str) == stage].copy().reset_index(drop=True)

        if len(stage_df) == 0:
            raise SystemExit("No rows for stage: %s" % stage)

        results, preds = run_stage(stage_df, args, device)
        all_results.extend(results)
        all_preds.extend(preds)

        pd.DataFrame(all_results).to_csv(outdir / "organic_geometry_GNN_intersection_results.csv", index=False)
        pd.DataFrame(all_preds).to_csv(outdir / "organic_geometry_GNN_intersection_predictions.csv", index=False)

    with open(outdir / "organic_geometry_GNN_intersection_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    elapsed = time.time() - t0
    print("=" * 80)
    print("DONE")
    print("elapsed_sec:", elapsed)
    print("saved:", outdir / "organic_geometry_GNN_intersection_results.csv")
    print("saved:", outdir / "organic_geometry_GNN_intersection_predictions.csv")
    print("saved:", outdir / "organic_geometry_GNN_intersection_config.json")


if __name__ == "__main__":
    main()
