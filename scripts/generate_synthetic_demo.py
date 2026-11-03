#!/usr/bin/env python3
"""Generate a deterministic, clearly synthetic demonstration dataset.

NEVER use these fabricated values as measured solvation free energies or as
quantum-chemical geometries. The fixture is only for end-to-end code testing.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

REPO = Path(__file__).resolve().parents[1]
SOLUTES = [
    'C', 'CC', 'CO', 'CCO', 'CC(=O)C', 'CC#N', 'COC', 'CCCl',
    'CCBr', 'CCF', 'CCS', 'O=C=O', 'CN', 'CCC', 'O=CO', 'C=O',
]
SOLVENTS = [('Methanol', 'CO'), ('Acetonitrile', 'CC#N'), ('Acetone', 'CC(=O)C')]
MANIFEST_COLUMNS = ['system_id', 'stage_mode', 'input_xyz', 'target',
                    'solute_smiles', 'solvent_smiles', 'solvent', 'n_solvent']
STAGES = ('initial', 'final')


def make_conformer(smiles: str, seed: int):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError(f'Invalid SMILES {smiles}')
    status = AllChem.EmbedMolecule(mol, randomSeed=seed, maxAttempts=50)
    if status != 0:
        raise RuntimeError(f'RDKit failed embedding {smiles}, seed={seed}')
    AllChem.UFFOptimizeMolecule(mol, maxIters=75)
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    return elements, coords - coords.mean(axis=0, keepdims=True)


def rotate(coords, rng):
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return coords @ q


def assemble(solute_smiles: str, solvent_smiles: str, sid: int, stage: str):
    rng = np.random.default_rng(12000 + sid * 37 + (stage == 'final') * 9)
    solute_elem, solute_xyz = make_conformer(solute_smiles, 222 + sid)
    solvent_elem, solvent_xyz = make_conformer(solvent_smiles, 721 + sid)
    solvent_blocks = []
    elems = list(solute_elem)
    positions = [solute_xyz]
    # This geometry is artificial; it is not a GFN2-xTB or ORCA result.
    for j in range(2):
        angle = 2 * math.pi * (j + 0.13 * (sid % 3)) / 2
        base_radius = 4.5 if stage == 'initial' else 3.9
        vec = np.array([base_radius * math.cos(angle),
                        base_radius * math.sin(angle), 0.75 * (j + 1)])
        block = rotate(solvent_xyz, rng) + vec
        # Avoid unrealistic near-overlap by increasing solvent translation.
        for _ in range(18):
            minimum = min(float(np.linalg.norm(block[:, None, :] - other[None, :, :], axis=-1).min())
                          for other in positions)
            if minimum > 1.5:
                break
            block += (vec / max(np.linalg.norm(vec), 1e-6)) * 0.4
        elems.extend(solvent_elem)
        positions.append(block)
        solvent_blocks.append(block)
    coords = np.concatenate(positions, axis=0)
    return elems, coords, len(solute_elem), len(solvent_elem)


def descriptor_values(columns, elements, coords, n_solute, n_solvent_atoms, sid, stage):
    """Synthetic *schema-compatible*, not chemically validated descriptor values."""
    rng = np.random.default_rng(91234 + sid * 100 + (stage == 'final'))
    solute = coords[:n_solute]
    solvent = coords[n_solute:]
    distances = np.linalg.norm(solute[:, None, :] - solvent[None, :, :], axis=-1).reshape(-1)
    heavy_a = np.array([e != 'H' for e in elements[:n_solute]])
    heavy_b = np.array([e != 'H' for e in elements[n_solute:]])
    dh = np.linalg.norm(solute[heavy_a, None, :] - solvent[None, heavy_b, :], axis=-1).reshape(-1)
    counts = Counter(elements)
    d = {name: float(rng.uniform(0.05, 1.5)) for name in columns}
    for i, name in enumerate(columns):
        if '_fraction' in name:
            d[name] = float(rng.uniform(0, 1))
        elif '_count' in name:
            d[name] = int(rng.integers(0, 5))
        elif name.endswith('_std'):
            d[name] = float(np.std(distances))
        elif name.endswith('_min'):
            d[name] = float(np.min(distances))
        elif name.endswith('_max'):
            d[name] = float(np.max(distances))
        elif name.endswith('_mean'):
            d[name] = float(np.mean(distances))
        elif name.endswith('_median'):
            d[name] = float(np.median(distances))
        elif name.endswith('_q10'):
            d[name] = float(np.quantile(distances, .1))
        elif name.endswith('_q25'):
            d[name] = float(np.quantile(distances, .25))
        elif name.endswith('_q75'):
            d[name] = float(np.quantile(distances, .75))
        elif name.endswith('_q90'):
            d[name] = float(np.quantile(distances, .9))
    d.update({
        'rg_total': float(np.sqrt(np.square(coords - coords.mean(axis=0)).sum(axis=1).mean())),
        'rg_solute': float(np.sqrt(np.square(solute - solute.mean(axis=0)).sum(axis=1).mean())),
        'rg_solvent': float(np.sqrt(np.square(solvent - solvent.mean(axis=0)).sum(axis=1).mean())),
        'dist_all_solute_solvent_min': float(np.min(distances)),
        'dist_heavy_solute_solvent_min': float(np.min(dh)),
        'n_atoms_total': len(elements),
        'n_atoms_solute': n_solute,
        'n_atoms_solvent_total': len(elements) - n_solute,
        'n_atoms_solvent_single': n_solvent_atoms,
        'n_solvent': 2,
        'charge': 0,
        'multiplicity': 1,
    })
    for sym in ('H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'):
        d[f'total_n_{sym}'] = counts.get(sym, 0)
    for cutoff in (2.5, 3., 3.5, 4., 4.5, 5.):
        key = str(cutoff).replace('.', 'p')
        d[f'contact_{key}_heavy_pair_count'] = int((dh < cutoff).sum())
        d[f'contact_{key}_heavy_pair_fraction'] = float((dh < cutoff).mean())
        d[f'contact_{key}_solute_heavy_atom_fraction'] = float(np.any(np.linalg.norm(
            solute[heavy_a, None, :] - solvent[None, heavy_b, :], axis=-1) < cutoff, axis=1).mean())
        d[f'contact_{key}_solvent_heavy_atom_fraction'] = float(np.any(np.linalg.norm(
            solute[heavy_a, None, :] - solvent[None, heavy_b, :], axis=-1) < cutoff, axis=0).mean())
    # Remaining 314 fields are placeholder features, not validated descriptors.
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='data/demo', help='Repository-relative output folder')
    ap.add_argument('--replace', action='store_true', help='Replace only a pre-existing generated demo folder')
    args = ap.parse_args()
    out = (REPO / args.output).resolve()
    if not out.is_relative_to(REPO):
        ap.error('Output directory must be inside the repository')
    if out.exists():
        if not args.replace:
            ap.error(f'Output exists; pass --replace explicitly: {out}')
        if not (out / 'SYNTHETIC_DEMO_MARKER.txt').exists():
            ap.error('Refusing to overwrite a non-synthetic dataset')
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / 'xyz').mkdir()
    schema = json.loads((REPO / 'data/descriptor_columns.json').read_text())
    assert len(schema) == 314
    assert [schema[i] for i in (58,121,284,313)] == [
        'rg_total', 'contact_5p0_solvent_heavy_atom_fraction',
        'solvent_molecule_min_dist_to_solute_min', 'solvent_molecule_within_5p0_fraction']

    manifest_rows, descriptor_rows = [], []
    for index, solute in enumerate(SOLUTES, start=1):
        solvent, solvent_smiles = SOLVENTS[(index - 1) % len(SOLVENTS)]
        # Arbitrary synthetic label, constant over input-geometry stages.
        target = round(-2.4 - 0.18 * index - 0.3 * ((index - 1) % 3), 6)
        sid = f'toy_{index:03d}'
        for stage in STAGES:
            elements, coords, n_solute, n_solvent_atoms = assemble(solute, solvent_smiles, index, stage)
            relative = f'{args.output.rstrip("/")}/xyz/{sid}_{stage}.xyz'
            xyz_path = REPO / relative
            with xyz_path.open('w') as f:
                f.write(f'{len(elements)}\n')
                f.write('SYNTHETIC demo only; NOT a quantum-optimized cluster\n')
                for e, pos in zip(elements, coords):
                    f.write(f'{e:<2} {pos[0]: .8f} {pos[1]: .8f} {pos[2]: .8f}\n')
            mr = dict(system_id=sid, stage_mode=stage, input_xyz=relative,
                      target=target, solute_smiles=solute, solvent_smiles=solvent_smiles,
                      solvent=solvent, n_solvent=2)
            manifest_rows.append(mr)
            dr = descriptor_values(schema, elements, coords, n_solute, n_solvent_atoms, index, stage)
            dr.update(dict(dataset_subset='SYNTHETIC_DEMO_ONLY', system_id=sid,
                           frame_id=f'{sid}_{stage}', stage=stage, stage_mode=stage,
                           solute_id=sid, solute_name='synthetic_demo',
                           solute_smiles=solute, solvent=solvent,
                           solvent_smiles=solvent_smiles, solvent_class='synthetic',
                           experimental_FE_kcal_mol=target, input_xyz=relative,
                           strict_no_fallback=1, target=target))
            descriptor_rows.append(dr)
    with (out / 'manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader(); writer.writerows(manifest_rows)
    with (out / 'descriptors.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=schema)
        writer.writeheader(); writer.writerows(descriptor_rows)
    (out / 'SYNTHETIC_DEMO_MARKER.txt').write_text(
        'SYNTHETIC DEMONSTRATION DATASET. NO EXPERIMENTAL TARGETS OR QUANTUM CALCULATIONS.\n'
        'Only for testing model loading, short training and file outputs.\n')
    (out / 'README.md').write_text(
        '# Synthetic demonstration data (not research data)\n\n'
        f'This folder contains {len(SOLUTES)} artificial molecular systems in {len(STAGES)} stages '
        f'({len(manifest_rows)} XYZ files), with **fabricated labels** and **schema-shaped '
        'synthetic descriptors**. Coordinates are RDKit-generated and loosely placed.\n\n'
        '**Do not use this data for scientific benchmarking, training-quality claims, or '
        'as experimental solvation free energies.** The only purpose is to check '
        'that the six training runners can load files, train briefly, and save outputs.\n\n'
        'The original 435-system experimental benchmark and 314-column descriptors '
        'must be obtained and redistributed separately after approval.\n')
    print(f'Wrote synthetic demo: {len(manifest_rows)} rows, {len(schema)} descriptor columns, '
          f'{len(list((out / "xyz").glob("*.xyz")))} XYZ files at {out}')

if __name__ == '__main__':
    main()
