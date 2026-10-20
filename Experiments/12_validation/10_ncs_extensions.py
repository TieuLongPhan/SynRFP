#!/usr/bin/env python
"""Registered supplementary analyses, separate from frozen validation gates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import vlib as L
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from synrfp_exp.config import expand_fingerprints, load_config
from synrfp_exp.data import dataset_table, reaction_table, rxn_key
from synrfp_exp.features import load
from synrfp_exp.splits import groups_for, make_split
from threadpoolctl import threadpool_limits

OUT = L.RUNS / 'ncs_extensions'
PANEL = ['synrfp_default', 'drfp', 'rdkit_struct', 'rdkit_diff', 'rxnfp_mlm']
ENDPOINTS = ['mech_2', 'uspto50k_b', 'ecreact_1', 'tpl_b']
POP = np.array([int(x).bit_count() for x in range(256)], dtype=np.int16)


def helper():
    spec = importlib.util.spec_from_file_location('canonical_clustering', L.HERE / '07_clustering.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def identity(s, undirected=False):
    a, _, b = str(s).partition('>>')
    if not _:
        parts = str(s).split('>')
        if len(parts) != 3:
            return None
        a, b = parts[0], parts[2]
    text = '|'.join(sorted((a, b))) if undirected else a + '>>' + b
    return hashlib.sha256(text.encode()).hexdigest()


def protocol(out, cfg, extra):
    h = helper()
    value = {'registered_protocol_sha256': L.sha256(L.HERE / 'COMPLETION_PROTOCOL.md'),
             'runner_sha256': L.sha256(Path(__file__)), 'config_sha256': L.sha256(L.policy_config_path('n1')),
             'environment': L.environment(), **extra}
    digest = h.stable_hash(value)
    path = out / 'protocol.json'
    if path.exists() and json.loads(path.read_text())['protocol_sha256'] != digest:
        raise RuntimeError(f'Protocol changed on resume: {path}')
    if not path.exists():
        h.atomic_json(path, {**value, 'protocol_sha256': digest})
    return digest


def overlap(cfg):
    out = OUT / 'overlap'
    protocol(out, cfg, {'matching': ['directed', 'reversal_equivalent'], 'tautomer_normalization': False})
    tables = {}
    for ds in cfg['datasets']:
        key = rxn_key(cfg, ds)
        if key not in tables:
            tables[key] = reaction_table(cfg, key)
    identities = {(key, rev): [identity(s, rev) for s in table.rxn]
                  for key, table in tables.items() for rev in (False, True)}
    pairs = []
    for rev in (False, True):
        for a in tables:
            A = Counter(identities[a, rev]); A.pop(None, None)
            for b in tables:
                B = Counter(identities[b, rev]); B.pop(None, None)
                shared = A.keys() & B.keys()
                pairs.append(dict(table_a=a, table_b=b, reversal_equivalent=rev,
                    rows_a=len(tables[a]), rows_b=len(tables[b]), unique_a=len(A), unique_b=len(B),
                    shared_unique=len(shared), rows_a_in_b=sum(A[k] for k in shared),
                    rows_b_in_a=sum(B[k] for k in shared)))
    pd.DataFrame(pairs).to_csv(out / 'cross_table.csv', index=False)
    splits = []
    for ds in cfg['datasets']:
        table = dataset_table(cfg, ds)
        rows = table.row.to_numpy()
        for seed in cfg['benchmark']['schemes']['cv']:
            code = make_split(cfg, ds, 'cv', int(seed))
            for rev in (False, True):
                keys = np.asarray(identities[rxn_key(cfg, ds), rev], dtype=object)[rows]
                for a, b in ((0, 1), (0, 2), (1, 2)):
                    A, B = set(keys[code == a]) - {None}, set(keys[code == b]) - {None}
                    splits.append(dict(dataset=ds, seed=seed, reversal_equivalent=rev,
                                       partition_a=a, partition_b=b, shared_unique=len(A & B),
                                       unique_a=len(A), unique_b=len(B)))
    pd.DataFrame(splits).to_csv(out / 'within_split.csv', index=False)
    metadata = []
    for key, table in tables.items():
        columns = [c for c in table.columns if any(s in str(c).lower() for s in ('patent', 'source', 'publication'))]
        metadata.append({'table': key, 'available_metadata_columns': ';'.join(columns),
                         'scope': 'available prepared metadata only; r_id is a row identifier'})
    pd.DataFrame(metadata).to_csv(out / 'metadata_coverage.csv', index=False)


def geometry(cfg, threads):
    h = helper(); out = OUT / 'geometry'
    panel = expand_fingerprints(cfg, PANEL)
    digest = protocol(out, cfg, {'datasets': ENDPOINTS, 'fingerprints': panel,
                                'seeds': [0, 1, 2], 'normalization': 'L2', 'rows': 'all_rows'})
    for ds_name in ENDPOINTS:
        for fp in panel:
            ds, X, valid, used, k, sources = h.prepare(cfg, ds_name, fp, panel, 'all_rows', {})
            X = normalize(X, copy=True)
            for seed in (0, 1, 2):
                path = out / ds_name / f'{fp}__s{seed}.json'
                signature = h.stable_hash({'protocol': digest, 'sources': sources, 'dataset': ds_name,
                                           'fingerprint': fp, 'seed': seed})
                if h.existing_complete(path, path.with_suffix('.npz'), signature):
                    continue
                method = 'kmeans' if ds_name in h.SMALL else 'minibatch_kmeans'
                from sklearn.cluster import KMeans
                estimator = KMeans(n_clusters=k, random_state=seed, **h.SMALL_PARAMS) if method == 'kmeans' else MiniBatchKMeans(n_clusters=k, random_state=seed, **h.LARGE_PARAMS)
                started = time.monotonic()
                with threadpool_limits(limits=threads):
                    labels = estimator.fit_predict(X)
                assign = path.with_suffix('.npz')
                h.atomic_npz(assign, labels=labels.astype(np.int32), row=ds.row.to_numpy(), used=used, valid=valid)
                h.atomic_json(path, {'dataset': ds_name, 'fingerprint': fp, 'seed': seed, 'method': method,
                    'normalization': 'L2', 'scores': h.scores(ds, cfg['datasets'][ds_name], labels, used),
                    'valid_fraction': float(valid.mean()), 'elapsed_seconds': time.monotonic() - started,
                    'assignment_sha256': L.sha256(assign), 'cell_signature': signature})
                print('geometry completed', ds_name, fp, seed, flush=True)


def nearest_members(X, centers, labels):
    """Actual reactions nearest to their own centre; deterministic row-order ties."""
    chosen = []
    for cid in np.unique(labels):
        idx = np.flatnonzero(labels == cid)
        distance = ((X[idx] - centers[int(cid)]) ** 2).sum(axis=1)
        chosen.append(int(idx[np.argmin(distance)]))
    return np.asarray(chosen, dtype=int)


def diversity_prefix(packed, n, first):
    """Greedy farthest-point Tanimoto selection, with stable row-order ties."""
    counts = POP[packed].sum(axis=1)
    nearest = np.full(len(packed), -np.inf)
    selected = np.zeros(len(packed), dtype=bool)
    result, current = [], int(first)
    for _ in range(n):
        result.append(current); selected[current] = True
        inter = POP[np.bitwise_and(packed, packed[current])].sum(axis=1)
        union = counts + counts[current] - inter
        sim = np.divide(inter, union, out=np.ones(len(packed), dtype=float), where=union > 0)
        nearest = np.maximum(nearest, sim)
        current = int(np.argmin(np.where(selected, np.inf, nearest)))
    return np.asarray(result, dtype=int)


def selection_scores(y, groups, selected):
    counts = Counter(y.tolist())
    rare = {k for k, v in counts.items() if v <= 20}
    covered = set(y[selected].tolist())
    all_groups = set(groups.tolist()) - {'<unparsable>'}
    covered_groups = set(groups[selected].tolist()) - {'<unparsable>'}
    return {'class_coverage': len(covered) / len(counts), 'classes_covered': len(covered),
            'rare_class_coverage': len(covered & rare) / len(rare) if rare else None,
            'rare_classes_total': len(rare), 'product_group_coverage': len(covered_groups) / len(all_groups) if all_groups else None,
            'representatives_actual': len(selected)}


def utility(cfg, threads):
    h = helper(); out = OUT / 'representative_selection'
    panel = expand_fingerprints(cfg, PANEL)
    digest = protocol(out, cfg, {'datasets': ['uspto50k_b', 'ecreact_1'], 'fingerprints': panel,
        'sample': 20000, 'sample_seed': 42, 'seeds': [0, 1, 2], 'budgets': [0.005, 0.01, 0.02, 0.05],
        'algorithm': 'MiniBatchKMeans', 'params': h.LARGE_PARAMS, 'n_clusters': 'ceil(sample_size*budget)',
        'rare_class_definition': 'sample_count<=20'})
    for ds_name in ('uspto50k_b', 'ecreact_1'):
        ds = dataset_table(cfg, ds_name)
        indices = np.sort(np.random.default_rng(42).choice(len(ds), min(20000, len(ds)), replace=False))
        rows = ds.row.to_numpy()[indices]; y = ds.y.to_numpy()[indices]
        groups = groups_for(cfg, ds_name)[indices]
        h.atomic_npz(out / f'{ds_name}__sample.npz', dataset_index=indices, reaction_row=rows)
        budgets = [(b, int(math.ceil(len(rows) * b))) for b in (0.005, 0.01, 0.02, 0.05)]
        baseline = load(cfg, rxn_key(cfg, ds_name), 'rdkit_struct')
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            random_order = rng.permutation(len(rows))
            diverse_paths = [out / ds_name / f'rdkit_struct_greedy__b{b}__s{seed}.json' for b, _ in budgets]
            diverse = None if all(p.exists() for p in diverse_paths) else diversity_prefix(baseline.X[rows], max(k for _, k in budgets), int(random_order[0]))
            for b, k in budgets:
                for fp, order in (('uniform_random', random_order), ('rdkit_struct_greedy', diverse)):
                    path = out / ds_name / f'{fp}__b{b}__s{seed}.json'
                    if path.exists():
                        continue
                    selected = order[:k]
                    h.atomic_npz(path.with_suffix('.npz'), selected_dataset_index=indices[selected], selected_reaction_row=rows[selected])
                    h.atomic_json(path, {'dataset': ds_name, 'fingerprint': fp, 'budget': b, 'seed': seed,
                        'representatives_requested': k, 'sample_rows': len(rows), 'protocol_sha256': digest,
                        **selection_scores(y, groups, selected)})
        for fp in panel:
            feature = load(cfg, rxn_key(cfg, ds_name), fp)
            X = feature.dense(rows)
            if not np.isfinite(X).all():
                raise RuntimeError(f'Nonfinite utility features: {ds_name}/{fp}')
            for seed in (0, 1, 2):
                for b, k in budgets:
                    path = out / ds_name / f'{fp}__b{b}__s{seed}.json'
                    if path.exists():
                        if json.loads(path.read_text())['protocol_sha256'] != digest:
                            raise RuntimeError(f'Changed utility protocol: {path}')
                        continue
                    started = time.monotonic()
                    estimator = MiniBatchKMeans(n_clusters=k, random_state=seed, **h.LARGE_PARAMS)
                    with threadpool_limits(limits=threads):
                        labels = estimator.fit_predict(X)
                        selected = nearest_members(X, estimator.cluster_centers_, labels)
                    h.atomic_npz(path.with_suffix('.npz'), selected_dataset_index=indices[selected], selected_reaction_row=rows[selected])
                    h.atomic_json(path, {'dataset': ds_name, 'fingerprint': fp, 'budget': b, 'seed': seed,
                        'representatives_requested': k, 'sample_rows': len(rows), 'protocol_sha256': digest,
                        'elapsed_seconds': time.monotonic() - started, 'valid_fraction': float(feature.ok[rows].mean()),
                        'selected_valid_fraction': float(feature.ok[rows[selected]].mean()),
                        **selection_scores(y, groups, selected)})
                    print('utility completed', ds_name, fp, seed, b, flush=True)


def bond_signature(rsmi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        parts = str(rsmi).split('>')
        if len(parts) != 3:
            return None, 'invalid_reaction'
        sides = [Chem.MolFromSmiles(parts[i], sanitize=False) for i in (0, 2)]
        if any(m is None for m in sides):
            return None, 'parse_failed'
        atoms, bonds = [], []
        for mol in sides:
            maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
            if not all(maps) or len(set(maps)) != len(maps):
                return None, 'missing_or_duplicate_maps'
            atoms.append({a.GetAtomMapNum(): (a.GetSymbol(), a.GetFormalCharge()) for a in mol.GetAtoms()})
            bonds.append({tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()))): str(b.GetBondType()) for b in mol.GetBonds()})
        changes = []
        for pair in bonds[0].keys() | bonds[1].keys():
            before, after = bonds[0].get(pair, 'NONE'), bonds[1].get(pair, 'NONE')
            if before != after:
                elements = sorted((atoms[0].get(m) or atoms[1][m])[0] for m in pair)
                changes.append(('bond', *elements, before, after))
        for m in atoms[0].keys() & atoms[1].keys():
            if atoms[0][m][1] != atoms[1][m][1]:
                changes.append(('charge', atoms[0][m][0], str(atoms[0][m][1]), str(atoms[1][m][1])))
        if not changes:
            return None, 'no_bond_or_charge_change'
        return json.dumps(sorted(changes)), 'ok'
    except Exception:
        return None, 'analysis_failed'


def bond_coherence(cfg):
    out = OUT / 'bond_coherence'; h = helper()
    protocol(out, cfg, {'dataset': 'mech_2', 'mapped_column': 'original_reactions',
        'signature': 'element_pair_bond_order_changes_and_mapped_atom_charge_changes', 'seeds': list(range(10))})
    ds = dataset_table(cfg, 'mech_2')
    raw = pd.read_csv(cfg['paths']['mech_raw'])
    analysed = [bond_signature(s) for s in raw.original_reactions]
    source_rows = ds.row.to_numpy()
    signatures = np.asarray([x[0] for x in analysed], dtype=object)[source_rows]
    status = np.asarray([x[1] for x in analysed], dtype=object)[source_rows]
    usable = status == 'ok'; groups = groups_for(cfg, 'mech_2')
    pd.DataFrame({'r_id': ds.r_id, 'status': status, 'bond_signature': signatures}).to_csv(out / 'signatures.csv', index=False)
    def pairs(vals):
        return sum(n * (n - 1) // 2 for n in Counter(vals).values())
    records = []
    for path in sorted((L.RUNS / 'clustering/n1/all_rows/mech_2').glob('*/*.json')):
        rec = json.loads(path.read_text())
        labels = np.load(path.with_suffix('.npz'))['labels']
        total = match = across = match_across = 0
        for cid in np.unique(labels[usable]):
            sel = usable & (labels == cid)
            sig, grp = signatures[sel].tolist(), groups[sel].tolist()
            n = len(sig); count = n * (n - 1) // 2
            total += count; match += pairs(sig)
            across += count - pairs(grp)
            match_across += pairs(sig) - pairs(list(zip(sig, grp)))
        records.append({'fingerprint': rec['fingerprint'], 'seed': rec['seed'], 'rows': len(ds),
            'signature_valid_rows': int(usable.sum()), 'signature_coverage': float(usable.mean()),
            'within_cluster_pairs': total, 'matching_signature_pairs': match,
            'signature_agreement': match / total if total else None,
            'different_product_group_pairs': across,
            'matching_signature_different_product_group_pairs': match_across,
            'signature_agreement_different_product_group': match_across / across if across else None})
    pd.DataFrame(records).to_csv(out / 'summary.csv', index=False)
    h.atomic_json(out / 'coverage.json', dict(Counter(status.tolist())))


def modern_prepare(cfg):
    out = OUT / 'modern_encoder'; h = helper()
    provenance = json.loads((L.RUNS / 'completion/crxnfp_provenance.json').read_text())
    protocol(out, cfg, {'datasets': ENDPOINTS, 'seeds': list(range(10)),
        'encoder': 'crxnfp', 'provenance': provenance, 'preprocessing': 'official', 'pooling': 'masked_mean'})
    directory = out / 'inputs'; directory.mkdir(parents=True, exist_ok=True)
    for ds_name in ENDPOINTS:
        key = rxn_key(cfg, ds_name)
        path = directory / f'{key}.csv.gz'
        if not path.exists():
            table = reaction_table(cfg, key)
            table[['rxn']].to_csv(path, index=False, compression='gzip')
    h.atomic_json(out / 'PREPARED.json', {'datasets': ENDPOINTS, 'inputs': {p.name: L.sha256(p) for p in directory.glob('*.csv.gz')}})


def shared_encoding_mask(masks):
    """A failed row is omitted for every representation in the matched arm."""
    sizes = {len(mask) for mask in masks.values()}
    if len(sizes) != 1 or not masks or any(np.asarray(mask).ndim != 1 for mask in masks.values()):
        raise RuntimeError('Shared-valid masks have inconsistent source alignment')
    return np.logical_and.reduce([np.asarray(mask, dtype=bool) for mask in masks.values()])


def reuse_shared_assignment(cfg, ds_name, fp, seed, ds, used, valid, feature_sha, dataset_sha=None):
    """Reuse an archived fit only when its inputs and matched rows are unchanged."""
    h = helper()
    method = 'kmeans' if ds_name in h.SMALL else 'minibatch_kmeans'
    path = L.RUNS / 'clustering/n1/shared_valid' / ds_name / fp / f'{method}__s{seed}.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text()); assignment = path.with_suffix('.npz')
    params = h.SMALL_PARAMS if ds_name in h.SMALL else h.LARGE_PARAMS
    if (record['dataset'] != ds_name or record['fingerprint'] != fp or record['seed'] != seed
            or record['input_policy'] != 'n1' or record['row_policy'] != 'shared_valid'
            or record['method'] != method or record['params'] != params
            or record['k'] != int(ds.y.nunique())):
        return None
    table = cfg['paths']['work'] / 'tables' / f'ds_{ds_name}.pkl'
    expected = [(table, dataset_sha or L.sha256(table)),
                (cfg['paths']['work'] / 'features' / rxn_key(cfg, ds_name) / f'{fp}.npz', feature_sha)]
    for source, digest in expected:
        archived = {value for name, value in record['source_sha256'].items()
                    if Path(name).resolve() == source.resolve()}
        if archived != {digest}:
            return None
    if not assignment.exists() or L.sha256(assignment) != record['assignment_sha256']:
        raise RuntimeError(f'Archived shared-valid assignment is missing or corrupted: {assignment}')
    with np.load(assignment, allow_pickle=False) as z:
        if (not np.array_equal(z['r_id'].astype(str), ds.r_id.astype(str).to_numpy())
                or not np.array_equal(z['source_row'], ds.row.to_numpy())
                or not np.array_equal(z['y'], ds.y.to_numpy())):
            raise RuntimeError(f'Archived shared-valid rows/targets differ: {assignment}')
        if not np.array_equal(z['used'], used) or not np.array_equal(z['valid'], valid):
            return None
        labels = z['labels'].copy()
    return labels, {'reused_assignment_file': str(assignment),
                    'reused_assignment_sha256': record['assignment_sha256'],
                    'original_fit_seconds': record['fit_seconds'],
                    'reuse_reason': 'same features, dataset, settings, initialization and shared-valid rows'}


def modern_score(cfg, threads):
    from sklearn.cluster import KMeans
    h = helper(); out = OUT / 'modern_encoder'
    panel = expand_fingerprints(cfg, PANEL)
    for ds_name in ENDPOINTS:
        ds = dataset_table(cfg, ds_name); rows = ds.row.to_numpy()
        feature = out / 'features' / f'{rxn_key(cfg, ds_name)}.npz'
        with np.load(feature) as z:
            X, valid = z['X'][rows], z['ok'][rows]
        if not np.isfinite(X).all():
            raise RuntimeError(f'Nonfinite crxnfp features: {ds_name}')
        metadata = json.loads(feature.with_suffix('.json').read_text())
        k = int(ds.y.nunique()); used = np.ones(len(ds), dtype=bool)
        dataset_sha = L.sha256(cfg['paths']['work'] / 'tables' / f'ds_{ds_name}.pkl')
        feature_sha = L.sha256(feature)
        if metadata.get('feature_sha256', feature_sha) != feature_sha:
            raise RuntimeError(f'crxnfp feature checksum differs from metadata: {feature}')
        for seed in range(10):
            path = out / 'clustering' / ds_name / f'crxnfp__s{seed}.json'
            signature = h.stable_hash({'feature': feature_sha, 'dataset': dataset_sha,
                'protocol': L.sha256(out / 'protocol.json'), 'seed': seed, 'adapter': L.sha256(L.HERE / 'modern_encoder.py')})
            if h.existing_complete(path, path.with_suffix('.npz'), signature):
                continue
            estimator = KMeans(n_clusters=k, random_state=seed, **h.SMALL_PARAMS) if ds_name in h.SMALL else MiniBatchKMeans(n_clusters=k, random_state=seed, **h.LARGE_PARAMS)
            started = time.monotonic()
            with threadpool_limits(limits=threads):
                labels = estimator.fit_predict(X)
            assign = path.with_suffix('.npz')
            h.atomic_npz(assign, labels=labels.astype(np.int32), source_row=rows,
                         r_id=ds.r_id.astype(str).to_numpy(dtype=str), y=ds.y.to_numpy(dtype=np.int32),
                         used=used, valid=valid)
            h.atomic_json(path, {'dataset': ds_name, 'fingerprint': 'crxnfp', 'seed': seed,
                'scores': h.scores(ds, cfg['datasets'][ds_name], labels, used), 'valid_fraction': float(valid.mean()),
                'input_policy': 'n1', 'row_policy': 'all_rows', 'n_rows': len(ds),
                'n_valid': int(valid.sum()), 'invalid_rows': int((~valid).sum()), 'n_scored': len(ds),
                'failure_policy': 'zero-vector placeholders included; full-row sensitivity arm, not evidence of successful encoding',
                'feature_metadata_file': str(feature.with_suffix('.json')),
                'feature_metadata_sha256': L.sha256(feature.with_suffix('.json')),
                'preprocessing_error_counts': metadata.get('preprocessing', {}).get('error_counts', {}),
                'inference_error_counts': metadata.get('inference', {}).get('error_counts', {}),
                'cell_signature': signature, 'assignment_sha256': L.sha256(assign),
                'elapsed_seconds': time.monotonic() - started, 'encoder_exposure': 'see crxnfp_provenance.json; not an independent-corpus claim'})

        # Keep the all-row arm above, and compare all six encoders on the exact
        # same successfully encoded rows. Existing five-encoder fits are reused
        # only when adding crxnfp leaves their row mask unchanged.
        masks, feature_files = {'crxnfp': np.asarray(valid, dtype=bool)}, {'crxnfp': feature}
        for fp in panel:
            cached = load(cfg, rxn_key(cfg, ds_name), fp)
            masks[fp] = np.asarray(cached.ok[rows], dtype=bool)
            feature_files[fp] = cfg['paths']['work'] / 'features' / rxn_key(cfg, ds_name) / f'{fp}.npz'
        shared = shared_encoding_mask(masks)
        if int(shared.sum()) < k:
            raise RuntimeError(f'Insufficient shared-valid rows for crxnfp comparison: {ds_name}')
        comparison = {'panel': [*panel, 'crxnfp'], 'n_rows': len(ds),
                      'n_shared_valid': int(shared.sum()), 'shared_valid_coverage': float(shared.mean()),
                      'invalid_encodings_by_fingerprint': {fp: int((~mask).sum()) for fp, mask in masks.items()},
                      'failure_policy': 'omit a row from every encoder when any encoder fails',
                      'resolution_rule': 'number of reference classes in the unchanged full dataset',
                      'source_feature_sha256': {fp: L.sha256(path) for fp, path in feature_files.items()},
                      'dataset_sha256': dataset_sha,
                      'modern_feature_metadata_file': str(feature.with_suffix('.json')),
                      'modern_feature_metadata_sha256': L.sha256(feature.with_suffix('.json'))}
        h.atomic_json(out / 'coverage' / f'{ds_name}.json', comparison)
        for fp in [*panel, 'crxnfp']:
            dense = None
            for seed in range(10):
                path = out / 'clustering_shared_valid' / ds_name / f'{fp}__s{seed}.json'
                signature = h.stable_hash({'protocol': 'modern_matched_valid_v1',
                    'comparison': comparison, 'fingerprint': fp, 'seed': seed,
                    'params': h.SMALL_PARAMS if ds_name in h.SMALL else h.LARGE_PARAMS})
                if h.existing_complete(path, path.with_suffix('.npz'), signature):
                    continue
                started = time.monotonic(); reuse = None
                if fp != 'crxnfp':
                    reuse = reuse_shared_assignment(cfg, ds_name, fp, seed, ds, shared,
                                                    masks[fp], comparison['source_feature_sha256'][fp], dataset_sha)
                elif shared.all():
                    original = out / 'clustering' / ds_name / f'crxnfp__s{seed}.npz'
                    with np.load(original, allow_pickle=False) as z:
                        reuse = (z['labels'].copy(), {'reused_assignment_file': str(original),
                            'reused_assignment_sha256': L.sha256(original),
                            'reuse_reason': 'shared-valid mask includes every row; identical all-row fit'})
                if reuse:
                    labels, reuse_info = reuse
                else:
                    if dense is None:
                        dense = X[shared] if fp == 'crxnfp' else load(cfg, rxn_key(cfg, ds_name), fp).dense(rows[shared])
                    if not np.isfinite(dense).all():
                        raise RuntimeError(f'Nonfinite matched features: {ds_name}/{fp}')
                    estimator = KMeans(n_clusters=k, random_state=seed, **h.SMALL_PARAMS) if ds_name in h.SMALL else MiniBatchKMeans(n_clusters=k, random_state=seed, **h.LARGE_PARAMS)
                    with threadpool_limits(limits=threads):
                        fitted = estimator.fit_predict(dense)
                    labels = np.full(len(ds), -1, dtype=np.int32); labels[shared] = fitted
                    reuse_info = {}
                assign = path.with_suffix('.npz')
                h.atomic_npz(assign, labels=labels.astype(np.int32), source_row=rows,
                             r_id=ds.r_id.astype(str).to_numpy(dtype=str), y=ds.y.to_numpy(dtype=np.int32),
                             used=shared, valid=masks[fp])
                h.atomic_json(path, {'dataset': ds_name, 'fingerprint': fp, 'seed': seed,
                    'input_policy': 'n1', 'row_policy': 'shared_valid_all_six',
                    'scores': h.scores(ds, cfg['datasets'][ds_name], labels, shared),
                    'n_rows': len(ds), 'n_valid': int(masks[fp].sum()),
                    'invalid_rows': int((~masks[fp]).sum()), 'n_scored': int(shared.sum()),
                    'valid_fraction': float(masks[fp].mean()), 'comparison': comparison,
                    'cell_signature': signature, 'assignment_sha256': L.sha256(assign),
                    'elapsed_seconds': time.monotonic() - started, **reuse_info})
            del dense


def stereo(cfg):
    from rdkit import Chem, RDLogger
    from synrfp_exp.features import compute_in_memory
    RDLogger.DisableLog('rdApp.*')
    h = helper(); out = OUT / 'stereo'
    protocol(out, cfg, {'datasets': ENDPOINTS, 'max_pairs_per_endpoint': 50, 'traversal_seed': 42,
        'perturbation': 'invert_first_assigned_product_tetrahedral_centre'})
    records, coverage = [], []
    for ds_name in ENDPOINTS:
        ds = dataset_table(cfg, ds_name); count = 0; inspected = 0
        for i in np.random.default_rng(42).permutation(len(ds)):
            inspected += 1; original = str(ds.iloc[i].rxn)
            fields = original.split('>')
            if len(fields) != 3: continue
            mol = Chem.MolFromSmiles(fields[2])
            if mol is None: continue
            centre = next((a for a in mol.GetAtoms() if a.GetChiralTag() in (
                Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)), None)
            if centre is None: continue
            centre.InvertChirality()
            alternative = '>'.join([fields[0], fields[1], Chem.MolToSmiles(mol, isomericSmiles=True)])
            base = Chem.MolFromSmiles(fields[2]); altered = Chem.MolFromSmiles(alternative.split('>')[2])
            if Chem.MolToSmiles(base, isomericSmiles=False) != Chem.MolToSmiles(altered, isomericSmiles=False):
                raise RuntimeError('Stereo perturbation changed constitutional identity')
            if Chem.MolToSmiles(base, isomericSmiles=True) == Chem.MolToSmiles(altered, isomericSmiles=True): continue
            records.append({'dataset': ds_name, 'r_id': str(ds.iloc[i].r_id), 'original': original, 'perturbed': alternative})
            count += 1
            if count == 50: break
        coverage.append({'dataset': ds_name, 'dataset_rows': len(ds), 'inspected_rows': inspected,
                         'valid_pairs': count, 'requested_pairs': 50})
    pairs = pd.DataFrame(records); pairs.to_csv(out / 'pairs.csv', index=False)
    pd.DataFrame(coverage).to_csv(out / 'coverage.csv', index=False)
    result = []
    for fp in expand_fingerprints(cfg, PANEL[:-1]):
        a = compute_in_memory(cfg, pairs.original.tolist(), fp)
        b = compute_in_memory(cfg, pairs.perturbed.tolist(), fp)
        for i, row in pairs.iterrows():
            result.append({'dataset': row.dataset, 'r_id': row.r_id, 'fingerprint': fp,
                'valid_original': bool(a.ok[i]), 'valid_perturbed': bool(b.ok[i]),
                'identical': bool(np.array_equal(a.X[i], b.X[i])),
                'l2_distance': float(np.linalg.norm(a.dense([i])[0] - b.dense([i])[0]))})
    pd.DataFrame(result).to_csv(out / 'cpu_results.csv', index=False)


def scaling(cfg, threads, cell=None):
    h = helper(); out = OUT / 'scaling'
    panel = expand_fingerprints(cfg, PANEL)
    digest = protocol(out, cfg, {'datasets': ['tpl_b', 'ecreact_1'], 'fingerprints': panel,
        'sizes': [10000, 50000, 100000, 'full'], 'sample_seed': 42, 'fit_seed': 0,
        'timing_repeats': [0, 1, 2], 'k': 128, 'threads': threads, 'params': h.LARGE_PARAMS})
    if cell is None:
        cells = [(ds, fp, n, rep) for ds in ('tpl_b', 'ecreact_1') for fp in panel
                 for n in ('10000', '50000', '100000', 'full') for rep in range(3)]
        np.random.default_rng(42).shuffle(cells)
        for ds, fp, n, rep in cells:
            path = out / ds / f'{fp}__n{n}__r{rep}.json'
            if path.exists(): continue
            command = [sys.executable, str(Path(__file__)), '--stage', 'scaling', '--threads', str(threads),
                       '--scaling-cell', ds, fp, n, str(rep)]
            subprocess.run(command, cwd=L.EXP, check=True)
        pd.DataFrame([json.loads(p.read_text()) for p in out.glob('*/*.json')]).to_csv(out / 'cells.csv', index=False)
        return
    ds_name, fp, size, rep = cell
    ds = dataset_table(cfg, ds_name)
    n = len(ds) if size == 'full' else min(int(size), len(ds))
    indices = np.sort(np.random.default_rng(42).permutation(len(ds))[:n])
    rows = ds.row.to_numpy()[indices]
    started = time.monotonic(); feature = load(cfg, rxn_key(cfg, ds_name), fp)
    X = feature.dense(rows); load_seconds = time.monotonic() - started
    estimator = MiniBatchKMeans(n_clusters=min(128, n), random_state=0, **h.LARGE_PARAMS)
    started = time.monotonic()
    with threadpool_limits(limits=threads): labels = estimator.fit_predict(X)
    seconds = time.monotonic() - started
    from synrfp_exp.metrics import clustering
    result = clustering(ds.y.to_numpy()[indices], labels)
    h.atomic_json(out / ds_name / f'{fp}__n{size}__r{rep}.json', {'dataset': ds_name,
        'fingerprint': fp, 'sample_size': n, 'requested_size': size, 'timing_repeat': int(rep),
        'fit_seconds': seconds, 'feature_load_seconds': load_seconds,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        'dense_array_bytes': int(X.nbytes), 'cached_array_bytes_per_reaction': feature.X.nbytes / len(feature.X),
        'encoding_coverage': float(feature.ok[rows].mean()), 'protocol_sha256': digest, **result})


def attribution_figure_records(root):
    """Read completed construction cells, excluding protocol/completion files."""
    columns = ('dataset', 'vocabulary', 'payload', 'hash_seed', 'seed',
               'ami', 'ari', 'mean_bits_set')
    records = []
    for path in sorted(root.glob('*/*/*__h*__s*.json')):
        record = json.loads(path.read_text())
        if record.get('status') != 'complete':
            continue
        missing = set(columns) - record.keys()
        if missing:
            raise RuntimeError(f'Completed attribution cell lacks {sorted(missing)}: {path}')
        if not all(np.isfinite(record[key]) for key in ('ami', 'ari', 'mean_bits_set')):
            raise RuntimeError(f'Completed attribution cell has nonfinite metrics: {path}')
        records.append({key: record[key] for key in columns})
    return records


def figures(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h = helper(); out = OUT / 'paper'; out.mkdir(parents=True, exist_ok=True)
    records = []
    for arm, root in [('primary', L.RUNS / 'clustering/n1/all_rows'),
                      ('shared_valid', L.RUNS / 'clustering/n1/shared_valid'),
                      ('L2_normalized', OUT / 'geometry'), ('modern', OUT / 'modern_encoder/clustering'),
                      ('modern_shared_valid', OUT / 'modern_encoder/clustering_shared_valid')]:
        for path in sorted(root.rglob('*.json')):
            r = json.loads(path.read_text())
            if 'scores' not in r or 'dataset' not in r:
                continue
            level = cfg['datasets'][r['dataset']]['label']
            records.append({'arm': arm, 'dataset': r['dataset'], 'fingerprint': r['fingerprint'],
                'seed': r['seed'], 'row_policy': r.get('row_policy'),
                'n_rows': r.get('n_rows'), 'n_scored': r.get('n_scored'),
                'invalid_encodings': r.get('invalid_rows'), 'failure_policy': r.get('failure_policy'),
                **r['scores'][level]})
    frame = pd.DataFrame(records)
    frame.to_csv(out / 'clustering_cells.csv', index=False)
    if len(frame):
        means = frame.groupby(['arm', 'dataset', 'fingerprint'])[['ami', 'ari']].agg(['mean', 'std', 'count'])
        means.to_csv(out / 'clustering_mean_std.csv')
        for metric in ('ami', 'ari'):
            data = frame[frame.arm == 'primary']
            table = data.pivot_table(index='dataset', columns='fingerprint', values=metric, aggfunc='mean')
            fig, ax = plt.subplots(figsize=(12, 6))
            table.plot.bar(ax=ax, width=0.85)
            ax.set_ylabel(metric.upper()); ax.set_xlabel(''); ax.legend(fontsize=7)
            ax.set_title('Canonical input; ten initialization seeds; known class-count resolution')
            fig.tight_layout()
            for extension in ('png', 'pdf'): fig.savefig(out / f'canonical_clustering_{metric}.{extension}', dpi=180)
            plt.close(fig)
            modern = frame[frame.arm == 'modern_shared_valid']
            if len(modern):
                table = modern.pivot_table(index='dataset', columns='fingerprint', values=metric, aggfunc='mean')
                fig, ax = plt.subplots(figsize=(12, 6))
                table.plot.bar(ax=ax, width=0.85)
                ax.set_ylabel(metric.upper()); ax.set_xlabel(''); ax.legend(fontsize=7)
                ax.set_title('Six encoders; identical valid rows; ten seeds; known class-count resolution')
                fig.tight_layout()
                for extension in ('png', 'pdf'): fig.savefig(out / f'modern_shared_valid_{metric}.{extension}', dpi=180)
                plt.close(fig)
        native = frame[(frame.arm == 'primary') & frame.seed.isin([0, 1, 2])]
        norm = frame[frame.arm == 'L2_normalized']
        paired = norm.merge(native, on=['dataset', 'fingerprint', 'seed'], suffixes=('_L2', '_native'))
        for m in ('ami', 'ari'):
            paired[f'{m}_L2_minus_native'] = paired[f'{m}_L2'] - paired[f'{m}_native']
        paired.to_csv(out / 'geometry_paired.csv', index=False)
        wide = frame[frame.arm == 'primary'].pivot(index=['dataset', 'seed'], columns='fingerprint', values='ami')
        syn = next(f for f in wide.columns if f.startswith('synrfp-'))
        margins = wide.drop(columns=syn).rsub(wide[syn], axis=0)
        margins.to_csv(out / 'primary_paired_ami_margins.csv')
    utility_records = [json.loads(p.read_text()) for p in (OUT / 'representative_selection').glob('*/*__s*.json')]
    if utility_records:
        utility_frame = pd.DataFrame(utility_records)
        utility_frame.to_csv(out / 'representative_selection_cells.csv', index=False)
        for dataset, data in utility_frame.groupby('dataset'):
            fig, ax = plt.subplots(figsize=(8, 5))
            for fp, rows in data.groupby('fingerprint'):
                grouped = rows.groupby('budget').class_coverage.agg(['mean', 'std'])
                ax.errorbar(grouped.index.to_numpy() * 100, grouped['mean'], yerr=grouped['std'], label=fp, marker='o')
            ax.set_xlabel('Representative budget (%)'); ax.set_ylabel('Reference-class coverage')
            ax.set_title(dataset + ': label-hidden selection'); ax.legend(fontsize=7)
            fig.tight_layout()
            for extension in ('png', 'pdf'): fig.savefig(out / f'representative_selection_{dataset}.{extension}', dpi=180)
            plt.close(fig)
    attribution = attribution_figure_records(L.RUNS / 'clustering_attribution/n1')
    if attribution:
        at = pd.DataFrame(attribution); at.to_csv(out / 'clustering_construction_cells.csv', index=False)
        contrasts = []
        for name, (positive, negative) in {'signed_minus_unsigned': ('multiset_signed', 'multiset_unsigned'),
                'multiset_minus_set': ('multiset_signed', 'set_signed'), 'change_minus_pooled': ('multiset_signed', 'pooled'),
                'change_minus_side_tagged_pooled': ('multiset_signed', 'pooled_side_tagged')}.items():
            keys = ['dataset', 'vocabulary', 'hash_seed', 'seed']
            p = at[at.payload == positive].merge(at[at.payload == negative], on=keys, suffixes=('_positive', '_negative'))
            for _, r in p.iterrows():
                contrasts.append({**{k: r[k] for k in keys}, 'contrast': name,
                                  'ami_difference': r.ami_positive - r.ami_negative, 'ari_difference': r.ari_positive - r.ari_negative})
        pd.DataFrame(contrasts).to_csv(out / 'clustering_construction_contrasts.csv', index=False)
    examples = []
    for path in (L.RUNS / 'cluster_coherence/n1/all_rows').rglob('*examples*.csv'):
        d = pd.read_csv(path); d['source_file'] = str(path.relative_to(L.RUNS)); examples.append(d)
    if examples:
        review = pd.concat(examples, ignore_index=True)
        review['assessor_1_shared_transformation'] = ''; review['assessor_2_shared_transformation'] = ''
        review['assessor_1_usefulness'] = ''; review['assessor_2_usefulness'] = ''
        review.to_csv(out / 'chemical_review_candidates.csv', index=False)
    h.atomic_json(out / 'ARTIFACTS.json', {p.name: L.sha256(p) for p in out.iterdir() if p.is_file() and p.name != 'ARTIFACTS.json'})


def publication_metadata():
    """Keep portable paper artifacts in the checkout, without local cache symlinks."""
    destination = L.HERE / 'publication_metadata'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for path in L.RUNS.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in ('.csv', '.json', '.md', '.png', '.pdf'):
            continue
        relative = path.relative_to(L.RUNS)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[str(relative)] = L.sha256(target)
    helper().atomic_json(destination / 'RELEASE_MANIFEST.json', {
        'files': manifest, 'source': 'local validation output tree',
        'scope': 'metadata, result tables, figures; cached arrays and model weights stay local'})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['overlap', 'geometry', 'utility', 'bond_coherence', 'modern_prepare', 'modern_score', 'figures', 'stereo', 'scaling', 'publication_metadata'], required=True)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--scaling-cell', nargs=4)
    args = ap.parse_args()
    cfg = load_config(L.policy_config_path('n1')); cfg['n_jobs'] = args.threads
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'overlap': overlap(cfg)
    elif args.stage == 'geometry': geometry(cfg, args.threads)
    elif args.stage == 'utility': utility(cfg, args.threads)
    elif args.stage == 'bond_coherence': bond_coherence(cfg)
    elif args.stage == 'modern_prepare': modern_prepare(cfg)
    elif args.stage == 'modern_score': modern_score(cfg, args.threads)
    elif args.stage == 'stereo': stereo(cfg)
    elif args.stage == 'scaling': scaling(cfg, args.threads, args.scaling_cell)
    elif args.stage == 'publication_metadata': publication_metadata()
    else: figures(cfg)


if __name__ == '__main__':
    main()
