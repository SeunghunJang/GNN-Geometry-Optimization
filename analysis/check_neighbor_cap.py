#!/usr/bin/env python3
"""Audit whether DimeNet++ max_num_neighbors=128 truncates any matched XYZ graph.

This script uses the exact DimeNet++ native_radius_graph eligibility rule:
Euclidean distance strictly below cutoff (default 5.0 Angstrom), no self edges,
with no lower distance limit. Counts are for *incoming neighbors per atom*.
Run on the full 435-system x 8-stage matched manifest before citing results.
No model is trained and no input data are modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

STAGES = ('initial', 'loose_t10', 'loose_t20', 'loose_t30',
          'tight_t10', 'tight_t20', 'tight_t30', 'final')


def path_mappings(items: list[str]) -> list[tuple[str, Path]]:
    mappings = []
    for text in items:
        if '=' not in text:
            raise ValueError(f'--path-map must be OLD_PREFIX=NEW_PREFIX: {text!r}')
        old, new = text.split('=', 1)
        if not old or not new:
            raise ValueError(f'--path-map prefix cannot be empty: {text!r}')
        mappings.append((old.rstrip('/'), Path(new).expanduser()))
    return mappings


def resolve_xyz(raw: str, manifest: Path, mappings: list[tuple[str, Path]]) -> Path:
    original = Path(str(raw)).expanduser()
    candidates = [original]
    if not original.is_absolute():
        candidates.insert(0, manifest.parent / original)
    src = str(original)
    for old, new in mappings:
        if src == old or src.startswith(old + '/'):
            remainder = src[len(old):].lstrip('/')
            candidates.insert(0, new / remainder)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    # Unresolved paths are reported together rather than silently excluded.
    return candidates[0]


def load_xyz(path: Path) -> np.ndarray:
    with path.open('r', encoding='utf-8') as f:
        first = f.readline().strip()
        try:
            count = int(first)
        except ValueError as ex:
            raise ValueError(f'{path}: not an XYZ atom-count header: {first!r}') from ex
        if count < 1:
            raise ValueError(f'{path}: XYZ requires at least one atom')
        f.readline()  # comment / metadata
        positions = np.empty((count, 3), dtype=np.float64)
        for atom in range(count):
            line = f.readline()
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f'{path}: missing/invalid atom line {atom + 1}')
            try:
                positions[atom] = tuple(float(q) for q in parts[1:4])
            except ValueError as ex:
                raise ValueError(f'{path}: invalid coordinates on atom line {atom + 1}') from ex
    if not np.isfinite(positions).all():
        raise ValueError(f'{path}: non-finite coordinates')
    return positions


def count_neighbors(pos: np.ndarray, cutoff: float, cap: int,
                    near_tolerance: float, chunk: int = 128) -> tuple[int, int, int, int]:
    """Return max count, atom centers exceeding cap, directed edges, near-cutoff pairs.

    Chunked pairwise distance avoids storing a full n-by-n matrix for clusters.
    The near-cutoff count is directed, excluding self pairs. Exact cutoff uses
    float64 coordinates; float32 PyTorch CUDA may differ at the boundary.
    """
    n = pos.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    near_pairs = 0
    cutoff_sq = cutoff * cutoff
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        delta = pos[start:stop, None, :] - pos[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', delta, delta)
        neighbors = d2 < cutoff_sq
        rows = np.arange(stop - start)
        cols = np.arange(start, stop)
        neighbors[rows, cols] = False
        counts[start:stop] = neighbors.sum(axis=1)
        if near_tolerance > 0:
            near = (d2 >= max(cutoff - near_tolerance, 0.0) ** 2) & (d2 <= (cutoff + near_tolerance) ** 2)
            near[rows, cols] = False
            near_pairs += int(near.sum())
    return int(counts.max()), int((counts > cap).sum()), int(counts.sum()), near_pairs


def validate_manifest(df: pd.DataFrame, allow_partial: bool) -> None:
    required = {'system_id', 'stage_mode', 'input_xyz'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing manifest columns: {sorted(required - set(df.columns))}')
    if df.duplicated(['system_id', 'stage_mode']).any():
        raise ValueError('Duplicate system_id/stage_mode entries in manifest')
    if not allow_partial:
        if len(df) != 3480:
            raise ValueError(f'Expected 3,480 rows (435 systems x 8 stages), got {len(df)}; --allow-partial is for diagnostics only')
        if set(df.stage_mode.astype(str)) != set(STAGES):
            raise ValueError('Matched manifest must contain exactly eight stage names')
        ids = set(df[df.stage_mode == STAGES[0]].system_id.astype(str))
        if len(ids) != 435:
            raise ValueError('Initial stage must contain 435 systems')
        for stage in STAGES:
            group = df[df.stage_mode == stage]
            if len(group) != 435 or set(group.system_id.astype(str)) != ids:
                raise ValueError(f'Matched stage membership differs: {stage}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True, type=Path, help='435 x 8 matched manifest with input_xyz paths')
    ap.add_argument('--cutoff', type=float, default=5.0, help='Strict spatial cutoff in Angstrom (default: 5.0)')
    ap.add_argument('--cap', type=int, default=128, help='Per-target-neighbor cap to audit (default: 128)')
    ap.add_argument('--near-tolerance', type=float, default=1e-5, help='Angstrom window around cutoff for numerical-boundary warning')
    ap.add_argument('--path-map', action='append', default=[], metavar='OLD_PREFIX=NEW_PREFIX',
                    help='Replace legacy XYZ path prefix; repeat for multiple prefixes')
    ap.add_argument('--report-json', type=Path, default=Path('run_outputs/neighbor_summary.json'))
    ap.add_argument('--report-csv', type=Path, default=Path('run_outputs/neighbor_counts.csv'))
    ap.add_argument('--allow-partial', action='store_true', help='Diagnostics only; result cannot certify all 3,480 structures')
    ap.add_argument('--no-fail-on-cap', action='store_true', help='Write results and exit zero even if cap is exceeded')
    args = ap.parse_args()
    if args.cutoff <= 0 or args.cap <= 0 or args.near_tolerance < 0:
        ap.error('cutoff and cap must be positive; near-tolerance must be nonnegative')
    df = pd.read_csv(args.manifest, low_memory=False)
    validate_manifest(df, args.allow_partial)
    mappings = path_mappings(args.path_map)
    resolved = [resolve_xyz(raw, args.manifest, mappings) for raw in df.input_xyz]
    missing = [(i, str(x)) for i, x in enumerate(resolved) if not x.is_file()]
    if missing:
        for i, path in missing[:12]:
            print(f'MISSING XYZ row {i}: {path}', file=sys.stderr)
        raise FileNotFoundError(f'{len(missing)} XYZ paths missing; fix manifest or --path-map. No partial PASS reported.')
    records = []
    for i, (row, path) in enumerate(zip(df.itertuples(index=False), resolved), start=1):
        pos = load_xyz(path)
        maximum, truncated_centers, edge_count, near_pairs = count_neighbors(
            pos, args.cutoff, args.cap, args.near_tolerance)
        records.append({'system_id': str(row.system_id), 'stage_mode': str(row.stage_mode),
                        'input_xyz': str(path), 'n_atoms': len(pos),
                        'max_neighbors_within_cutoff': maximum,
                        'centers_exceeding_cap': truncated_centers,
                        'directed_edges_uncapped': edge_count,
                        'directed_pairs_near_cutoff': near_pairs})
        if i % 500 == 0:
            print(f'Scanned {i}/{len(df)} XYZ structures', flush=True)
    result = pd.DataFrame(records)
    exceeded = result[result.centers_exceeding_cap > 0]
    top = result.loc[result.max_neighbors_within_cutoff.idxmax()]
    by_stage = {str(stage): {'structures': len(group),
                             'max_neighbors': int(group.max_neighbors_within_cutoff.max()),
                             'structures_exceeding_cap': int((group.centers_exceeding_cap > 0).sum())}
                for stage, group in result.groupby('stage_mode')}
    summary = {
        'dataset': '435-system, eight-stage matched benchmark',
        'full_matched_manifest_validated': not args.allow_partial,
        'structures_scanned': len(result), 'cutoff_A': args.cutoff,
        'strict_distance_rule': 'other atom at distance < cutoff (self excluded)',
        'max_num_neighbors': args.cap,
        'largest_atom_count': int(result.n_atoms.max()),
        'maximum_neighbors_observed': int(top.max_neighbors_within_cutoff),
        'max_neighbors_structure_id': str(top.system_id),
        'max_neighbors_stage': str(top.stage_mode),
        'structures_exceeding_cap': len(exceeded),
        'atom_centers_exceeding_cap': int(result.centers_exceeding_cap.sum()),
        'directed_pairs_near_cutoff': int(result.directed_pairs_near_cutoff.sum()),
        'near_cutoff_tolerance_A': args.near_tolerance,
        'floating_point_caveat': 'Calculated in NumPy float64; PyTorch float32/CUDA edge inclusion can differ for boundary pairs.',
        'by_stage': by_stage,
    }
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.report_csv, index=False)
    args.report_json.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f"AUDIT: {len(result)} XYZ structures, maximum neighbors = {summary['maximum_neighbors_observed']}, "
          f"structures exceeding {args.cap} = {len(exceeded)}")
    print(f'WROTE {args.report_json} and {args.report_csv}')
    if args.allow_partial:
        print('WARNING: partial diagnostic only; this does not certify the complete matched benchmark')
    if summary['directed_pairs_near_cutoff']:
        print(f"WARNING: {summary['directed_pairs_near_cutoff']} directed pairs lie near the cutoff; check PyTorch precision")
    if len(exceeded):
        print('WARNING: cap is active; graph connectivity may depend on native neighbor selection.')
        return 0 if args.no_fail_on_cap else 2
    print('PASS: no atom exceeds the specified neighbor cap in the scanned structures')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
