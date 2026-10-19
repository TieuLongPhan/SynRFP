"""Bounded, label-blind change/context audit of existing hashed token multisets.

This is a representation diagnostic, not a learned benchmark or an exact
chemical identity algorithm. Frozen experiments and their outputs are readonly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import heapq
import importlib.util
import json
from numbers import Integral
import os
from pathlib import Path
import re
import sys
import time

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = Path.home() / '.local/share/synrfp-runs/ncs_evidence_v1/token_audit'
EXPORT = HERE / 'publication_metadata/ncs_evidence_v1/token_audit'
SCHEMA = 'synrfp-bounded-token-information-v1'
SAMPLE_ROWS = 128
SOURCES = (
    ('all_endpoint_factorial_full_v1', (
        'bh_classic', 'suzuki_hte', 'ncs_bh', 'schneider_u', 'schneider_b',
        'tpl_u', 'tpl_b', 'uspto_50k_u', 'uspto_50k_b', 'syntemp', 'ecreact', 'mech')),
    ('cross_task_factorial_v1', (
        'feasibility', 'reagent', 'condition', 'gdb7', 'cyclo', 'snar', 'e2',
        'sn2', 'e2sn2', 'b97xd3', 'rdb7', 'rgd1')),
)
PLAN_SHA = {
    'all_endpoint_factorial_full_v1': '9bd10af8dee274984c50f6ba1cd661f7a4e028655555d5f18d7d26d198703e6b',
    'cross_task_factorial_v1': '077b5065a96c1073b6f0cba91814f9d842d804fc02097f7166fa210c70f40bf9',
}
SETTINGS = tuple((tokenizer, radius) for tokenizer in ('wl', 'morgan', 'path')
                 for radius in (2, 3))
MODES = ('delta', 'union', 'delta_union', 'roles')
PROTOCOL = {
    'schema': SCHEMA, 'sample_rows_per_unique_source': SAMPLE_ROWS,
    'sampling': 'smallest SHA256(schema, pool, source, original position, row_id); no target or result access',
    'source_scope': {pool: list(keys) for pool, keys in SOURCES},
    'population': 'all source rows are eligible for sampling; selected native eligibility refusals are retained',
    'settings': [{'tokenizer': t, 'radius': r} for t, r in SETTINGS],
    'modes': list(MODES), 'sketch': 'native parity', 'bits': 2048, 'seed': 1,
    'composite_blocks': {'delta_union': ['signed_delta:1024', 'union:1024'],
                         'roles': ['signed_delta:1024', 'shared:1024']},
    'algebra': 'D=P-R; U=P+R; S=min(R,P); U=abs(D)+2*S',
    'role_semantics': 'shared local token IDs, not mapped-atom retention or reagent roles',
    'equalities': 'full Python tuple/byte equality; SHA256 digests index receipts only',
    'collision_scope': 'observed within-source pairs of successfully parsed sampled rows only',
    'collision_levels': ['full count signature', 'unweighted signed/unsigned support', 'packed sketch bytes'],
    'identity_boundary': 'native token IDs are already hashes; equality is not proof of chemical identity',
    'labels_read': False, 'models_fit': False, 'stereo_features': False,
    'source_deduplication': 'only identical table SHA256, parser profile and input-column contract',
    'failure_policy': 'only native source parse ValueError and original eligibility become row refusals; other exceptions stop',
    'resources': {'processes': 1, 'threads': 1, 'intended_cpu_quota': 2, 'intended_memory_max_gib': 4},
}
_F = None
_B = None


class ContractError(ValueError):
    """A frozen population, source, algebra or information contract changed."""


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if not rows:
        raise ContractError('Empty aggregate output')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def native():
    global _F
    if _F is None:
        _F = module('ncs31_frozen20_token_encoder',
                    HERE.parent / '20_focused_factorial/encode_factorial.py')
    return _F


def initialize():
    from rdkit import RDLogger
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    RDLogger.DisableLog('rdApp.*')


def natural(counts):
    if not isinstance(counts, (dict, Counter)):
        raise ContractError('Token multiset must be a mapping')
    out = Counter()
    for token, count in counts.items():
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise ContractError('Native token IDs must be integers')
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ContractError('Token counts must be natural integers')
        if count:
            out[int(token)] = int(count)
    return out


def occurrence_oracle(reactant, product):
    """Independent occurrence-set construction; no Counter subtraction/min use."""
    reactant, product = natural(reactant), natural(product)
    if sum(reactant.values()) + sum(product.values()) > 1_000_000:
        raise ContractError('Bounded occurrence oracle exceeded one million occurrences')
    r = {(token, occurrence) for token, count in reactant.items() for occurrence in range(count)}
    p = {(token, occurrence) for token, count in product.items() for occurrence in range(count)}
    lost = Counter(token for token, _ in r - p)
    gained = Counter(token for token, _ in p - r)
    shared = Counter(token for token, _ in r & p)
    return lost, gained, shared


def assert_algebra(reactant, product, delta, union, shared):
    """Check full-count identity and both reversible coordinate systems."""
    r, p = natural(reactant), natural(product)
    u, s = natural(union), natural(shared)
    d = {}
    for token, count in delta.items():
        if (isinstance(token, bool) or not isinstance(token, Integral) or
                isinstance(count, bool) or not isinstance(count, Integral)):
            raise ContractError('Signed delta requires integer IDs and counts')
        if count:
            d[int(token)] = int(count)
    lost, gained, independent_s = occurrence_oracle(r, p)
    independent_d = {t: gained.get(t, 0) - lost.get(t, 0)
                     for t in lost.keys() | gained.keys()}
    if d != independent_d or s != independent_s:
        raise ContractError('Native delta/shared disagree with occurrence oracle')
    keys = r.keys() | p.keys() | d.keys() | u.keys() | s.keys()
    for token in keys:
        dd, uu, ss = d.get(token, 0), u.get(token, 0), s.get(token, 0)
        if uu != abs(dd) + 2 * ss:
            raise ContractError('U != abs(D) + 2S')
        if (uu - abs(dd)) % 2 or (uu - abs(dd)) // 2 != ss:
            raise ContractError('D,U do not recover S')
        if ((uu - dd) % 2 or (uu + dd) % 2 or
                (uu - dd) // 2 != r.get(token, 0) or
                (uu + dd) // 2 != p.get(token, 0)):
            raise ContractError('D,U do not recover original counts')
        if ss + max(-dd, 0) != r.get(token, 0) or ss + max(dd, 0) != p.get(token, 0):
            raise ContractError('D,S do not recover original counts')
    return lost, gained, independent_s


def signatures(delta, union, shared, mode):
    def counts(value):
        return tuple(sorted((int(k), int(v)) for k, v in value.items() if v))
    ds = tuple(sorted((int(k), 1 if v > 0 else -1) for k, v in delta.items() if v))
    us = tuple(sorted(int(k) for k, v in union.items() if v))
    ss = tuple(sorted(int(k) for k, v in shared.items() if v))
    if mode == 'delta':
        return counts(delta), ds
    if mode == 'union':
        return counts(union), us
    if mode == 'delta_union':
        return (counts(delta), counts(union)), (ds, us)
    if mode == 'roles':
        return (counts(delta), counts(shared)), (ds, ss)
    raise ContractError('Unknown native mode')


def choose2(n):
    return n * (n - 1) // 2


def equality_counts(records):
    """Records are full-count tuple, support tuple and full packed bytes.

    Equality never relies on a digest. Difference counts isolate count-support
    information loss from subsequent sketch collisions in this finite sample.
    """
    full, support, sketches = Counter(), Counter(), defaultdict(list)
    full_images, support_images = {}, {}
    for count_signature, support_signature, packed in records:
        if not isinstance(packed, bytes):
            raise ContractError('Collision oracle requires complete sketch bytes')
        for key, images in ((count_signature, full_images), (support_signature, support_images)):
            if key in images and images[key] != packed:
                raise ContractError('One complete signature produced unequal sketches')
            images[key] = packed
        full[count_signature] += 1
        support[support_signature] += 1
        sketches[packed].append((count_signature, support_signature))
    same_full = sum(choose2(v) for v in full.values())
    same_support = sum(choose2(v) for v in support.values())
    same_sketch = sum(choose2(len(v)) for v in sketches.values())
    diff_full = sum(choose2(len(v)) - sum(choose2(n) for n in Counter(x[0] for x in v).values())
                    for v in sketches.values())
    diff_support = sum(choose2(len(v)) - sum(choose2(n) for n in Counter(x[1] for x in v).values())
                       for v in sketches.values())
    total = choose2(len(records))
    if not (same_full <= same_support <= same_sketch <= total):
        raise ContractError('Equivalent full/support signatures produced unequal sketches')
    return {'valid_rows': len(records), 'observed_pairs': total,
            'equal_full_count_pairs': same_full, 'equal_support_pairs': same_support,
            'equal_sketch_pairs': same_sketch,
            'equal_sketch_different_full_count_pairs': diff_full,
            'equal_sketch_different_support_pairs': diff_support,
            'different_full_count_pairs': total - same_full,
            'different_support_pairs': total - same_support,
            'sketch_equality_rate_given_different_full_count': diff_full / (total - same_full) if total > same_full else None,
            'sketch_equality_rate_given_different_support': diff_support / (total - same_support) if total > same_support else None}


def reaction(rsmi, profile):
    global _B
    if not isinstance(rsmi, str):
        raise ValueError('Non-string source reaction')
    f = native()
    if profile == 'classification':
        return f.Reaction.from_rsmi(rsmi, agents=f.AGENTS)
    if profile != 'yield':
        raise ContractError('Unknown frozen parser profile')
    if rsmi.count('>>') != 1:
        raise ValueError('Expected exactly one explicit molecular-side delimiter')
    left, right = rsmi.split('>>')
    if _B is None:
        _B = module('ncs31_frozen13_explicit_sides', HERE.parent / '13_yield/bh_features.py')
    _B._strict_molecule(left)
    _B._strict_molecule(right)
    return f.Reaction.from_graph(_B._graph(left), _B._graph(right))


def boolean(value):
    if value in ('True', 'true', '1'):
        return True
    if value in ('False', 'false', '0'):
        return False
    raise ContractError('Invalid native eligibility field')


def input_columns(source):
    """Only request input-side columns; target columns are not materialized."""
    import pandas as pd
    header = pd.read_csv(source['table'], nrows=0).columns.tolist()
    selected = source.get('input_columns')
    if selected is None:
        selected = {'left': 'left', 'right': 'right'} if {'left', 'right'} <= set(header) else {'reaction': 'rxn'}
    if (set(selected) - {'reaction', 'left', 'right', 'eligible'} or
            any(v not in ('rxn', 'reaction', 'aam', 'left', 'right', 'eligible') for v in selected.values())):
        raise ContractError('A non-input column was requested')
    if not ('reaction' in selected or {'left', 'right'} <= set(selected)):
        raise ContractError('Source has no declared molecular input')
    if any(v not in header for v in selected.values()) or 'row_id' not in header:
        raise ContractError('Source input or original row identity missing')
    return dict(selected)


def select_sample(source, pool, key, sample_rows=SAMPLE_ROWS):
    import pandas as pd
    columns = input_columns(source)
    names = list(dict.fromkeys(['row_id', *columns.values()]))
    retained = []
    total = 0
    reader = pd.read_csv(source['table'], usecols=names, dtype=str,
                         keep_default_na=False, chunksize=8192)
    try:
        for frame in reader:
            for offset, values in enumerate(frame.itertuples(index=False, name=None)):
                raw = dict(zip(frame.columns, values))
                position = total + offset
                identity = [SCHEMA, pool, key, position, raw['row_id']]
                rank = int(digest(identity), 16)
                if len(retained) == sample_rows and rank >= -retained[0][0]:
                    continue
                rsmi = (raw[columns['left']] + '>>' + raw[columns['right']]
                        if {'left', 'right'} <= set(columns) else raw[columns['reaction']])
                if 'reaction' in columns and {'left', 'right'} <= set(columns) and raw[columns['reaction']] != rsmi:
                    raise ContractError('Declared explicit sides disagree with reaction field')
                eligible = boolean(raw[columns['eligible']]) if 'eligible' in columns else True
                item = (-rank, position, raw['row_id'], rsmi, eligible)
                if len(retained) < sample_rows:
                    heapq.heappush(retained, item)
                else:
                    heapq.heapreplace(retained, item)
            total += len(frame)
    finally:
        reader.close()
    if total != int(source['n']):
        raise ContractError('Original source row count changed')
    return sorted(retained, key=lambda item: item[1])


def pin_sources():
    sources, pins = [], {}
    upstream = {}
    for pool, keys in SOURCES:
        root = Path.home() / '.local/share/synrfp-runs' / pool
        plan_path, seal_path = root / 'plan.json', root / 'plan_seal.json'
        plan_sha = sha(plan_path)
        if plan_sha != PLAN_SHA[pool] or json.loads(seal_path.read_text())['sha256'] != plan_sha:
            raise ContractError('Frozen parent plan changed: ' + pool)
        plan = json.loads(plan_path.read_text())
        if Path(plan['root']).resolve() != root.resolve():
            raise ContractError('Parent plan has unexpected run root')
        upstream[pool] = plan
        pins[str(plan_path)] = plan_sha
        pins[str(seal_path)] = sha(seal_path)
        for key in keys:
            if key not in plan['sources']:
                raise ContractError('Required frozen source missing: ' + key)
            source = plan['sources'][key]
            if source['parser_profile'] not in ('classification', 'yield'):
                raise ContractError('Unknown frozen source parser profile: ' + key)
            table = str(Path(source['table']).resolve(strict=True))
            expected = plan.get('pinned_files', {}).get(source['table'])
            if expected is None:
                expected = plan.get('pinned_files', {}).get(table)
            if expected is None:
                raise ContractError('Frozen source table lacks a parent pin: ' + key)
            actual = sha(table)
            if actual != expected:
                raise ContractError('Frozen source table changed: ' + key)
            pins[table] = actual
            selected = input_columns(source)
            sources.append({'pool': pool, 'key': key, 'source': source,
                            'table_sha256': actual, 'input_columns': selected})
    paths = [HERE / 'token_audit.py', HERE / 'test_token_audit.py',
             HERE.parent / '20_focused_factorial/encode_factorial.py',
             HERE.parent / '13_yield/bh_features.py', HERE.parent / 'synrfp_exp/features.py']
    paths.extend(sorted((REPO / 'synrfp').rglob('*.py')))
    for path in paths:
        path = path.resolve(strict=True)
        actual = sha(path)
        for plan in upstream.values():
            expected = plan.get('pinned_files', {}).get(str(path))
            if expected is not None and actual != expected:
                raise ContractError('Frozen encoding source changed: ' + str(path))
        pins[str(path)] = actual
    # Alias only byte-identical declared source populations. Different table
    # bytes remain separate even if some rows happen to share chemistry.
    unique = {}
    for item in sources:
        source = item['source']
        identity = (item['table_sha256'], source['parser_profile'], canonical(item['input_columns']))
        alias = item['pool'] + '/' + item['key']
        if identity in unique:
            unique[identity]['aliases'].append(alias)
        else:
            item['aliases'] = [alias]
            unique[identity] = item
    return list(unique.values()), pins


def verify_pins(pins):
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ContractError('Pinned input changed during audit: ' + path)


def row_statistics(r, p, d, u, s):
    total = sum(u.values())
    shared_mass = sum(s.values())
    delta_mass = sum(abs(v) for v in d.values())
    return {'reactant_occurrences': sum(r.values()), 'product_occurrences': sum(p.values()),
            'union_occurrences': total, 'delta_absolute_occurrences': delta_mass,
            'lost_occurrences': sum(-v for v in d.values() if v < 0),
            'gained_occurrences': sum(v for v in d.values() if v > 0),
            'shared_occurrences': shared_mass,
            'cancellation_fraction': 2 * shared_mass / total if total else 0.0,
            'change_mass_fraction': delta_mass / total if total else 0.0,
            'union_token_ids': len(u), 'delta_token_ids': len(d), 'shared_token_ids': len(s),
            'lost_token_ids': sum(v < 0 for v in d.values()),
            'gained_token_ids': sum(v > 0 for v in d.values()),
            'partial_shared_changed_token_ids': len(d.keys() & s.keys()),
            'empty_delta': int(not d), 'empty_shared': int(not s)}


def summarize(stats):
    if not stats:
        return {'valid_rows': 0, **{key + '_mean': None for key in row_statistics(Counter(), Counter(), Counter(), Counter(), Counter())}}
    return {'valid_rows': len(stats), **{key + '_mean': sum(row[key] for row in stats) / len(stats) for key in stats[0]}}


def audit_source(item, root, pins_sha):
    f = native()
    source, key = item['source'], item['key']
    name = item['pool'] + '__' + key
    folder = root / 'sources' / name
    complete_path = folder / 'COMPLETE.json'
    if complete_path.exists():
        receipt = json.loads(complete_path.read_text())
        if receipt.get('pins_sha256') != pins_sha or receipt.get('protocol_sha256') != digest(PROTOCOL):
            raise ContractError('Changed source checkpoint contract')
        verify_pins(receipt['outputs'])
        return receipt
    folder.mkdir(parents=True, exist_ok=True)
    sample = select_sample(source, item['pool'], key)
    selected = [{'source_position': p, 'row_id_sha256': digest(row_id),
                 'selection_sha256': f'{-rank:064x}', 'original_eligible': eligible}
                for rank, p, row_id, _, eligible in sample]
    write(folder / 'sample.json', {'schema': SCHEMA, 'source': key,
                                   'table_sha256': item['table_sha256'],
                                   'selection_inputs': 'source position and original row_id only', 'rows': selected})
    tokenizers = {name: f._tokenizer(name, f.NODE_ATTRS, f.EDGE_ATTRS) for name in f.TOKENIZERS}
    stats = defaultdict(list)
    equality = defaultdict(list)
    failures = Counter()
    row_count = 0
    rows_path = folder / 'rows.jsonl'
    temporary = rows_path.with_name(rows_path.name + '.tmp')
    started = time.monotonic()
    with temporary.open('w') as stream:
        for sample_receipt, (_, position, _, rsmi, eligible) in zip(selected, sample):
            identity = {'source': key, **sample_receipt}
            if not eligible:
                failure = 'original_eligibility'
                failures[failure] += 1
                stream.write(canonical({**identity, 'status': 'refused', 'stage': failure}) + '\n')
                continue
            try:
                rxn = reaction(rsmi, source['parser_profile'])
            except ValueError as error:
                # Do not persist error text: a parser can include source SMILES.
                failures['source_parse'] += 1
                stream.write(canonical({**identity, 'status': 'refused', 'stage': 'source_parse',
                                        'error_type': type(error).__name__}) + '\n')
                continue
            for tokenizer, radius in SETTINGS:
                r = natural(tokenizers[tokenizer].tokens_graph(rxn.reactant, radius))
                p = natural(tokenizers[tokenizer].tokens_graph(rxn.product, radius))
                d, u, s = f.token_algebra(r, p)
                assert_algebra(r, p, d, u, s)
                values = f.encode_tokens(r, p, 'parity')
                metrics = row_statistics(r, p, d, u, s)
                stats[(tokenizer, radius)].append(metrics)
                signatures_receipt = {}
                for mode in MODES:
                    vector = values[mode]
                    if vector.dtype.name != 'uint8' or vector.shape != (256,):
                        raise ContractError('Native sketch no longer has 2048 packed bits')
                    full, support = signatures(d, u, s, mode)
                    packed = vector.tobytes()
                    equality[(tokenizer, radius, mode)].append((full, support, packed))
                    signatures_receipt[mode] = {'full_count_receipt_sha256': digest(full),
                                               'support_receipt_sha256': digest(support),
                                               'packed_sketch_sha256': hashlib.sha256(packed).hexdigest(),
                                               'bits_set': sum(int(byte).bit_count() for byte in packed)}
                stream.write(canonical({**identity, 'status': 'ok', 'tokenizer': tokenizer,
                                        'radius': radius, **metrics, 'receipt_hashes': signatures_receipt}) + '\n')
                row_count += 1
    temporary.replace(rows_path)
    aggregate, collisions = [], []
    for tokenizer, radius in SETTINGS:
        prefix = {'source_pool': item['pool'], 'source': key, 'aliases': '|'.join(item['aliases']),
                  'tokenizer': tokenizer, 'radius': radius, 'sampled_rows': len(sample),
                  'original_eligibility_refusals': failures['original_eligibility'],
                  'parse_refusals': failures['source_parse']}
        aggregate.append({**prefix, **summarize(stats[(tokenizer, radius)])})
        for mode in MODES:
            collisions.append({**prefix, 'mode': mode, **equality_counts(equality[(tokenizer, radius, mode)])})
    csv_write(folder / 'token_information.csv', aggregate)
    csv_write(folder / 'observed_equalities.csv', collisions)
    outputs = {str(path): sha(path) for path in [folder / 'sample.json', rows_path,
               folder / 'token_information.csv', folder / 'observed_equalities.csv']}
    receipt = {'schema': SCHEMA, 'status': 'complete', 'created_utc': now(),
               'source': key, 'aliases': item['aliases'], 'source_pool': item['pool'],
               'pins_sha256': pins_sha, 'protocol_sha256': digest(PROTOCOL),
               'table_sha256': item['table_sha256'], 'sampled_rows': len(sample),
               'successful_setting_rows': row_count, 'refusals': dict(failures),
               'elapsed_seconds': time.monotonic() - started, 'outputs': outputs}
    write(complete_path, receipt)
    return receipt


def completed(root=ROOT):
    root = Path(root).resolve()
    receipt = json.loads((root / 'COMPLETE.json').read_text())
    if receipt.get('schema') != SCHEMA or receipt.get('phase') != 'complete':
        raise ContractError('Wrong completion receipt')
    verify_pins(receipt['outputs'])
    return receipt


def run(root=ROOT, export=EXPORT):
    initialize()
    root, export = Path(root).resolve(), Path(export).resolve()
    root.mkdir(parents=True, exist_ok=True)
    export.mkdir(parents=True, exist_ok=True)
    if (root / 'COMPLETE.json').exists():
        value = completed(root)
        verify_pins(value['pins'])
        return value
    started = time.monotonic()
    write(root / 'status.json', {'schema': SCHEMA, 'phase': 'running', 'stage': 'pinning_inputs', 'updated_utc': now()})
    sources, pins = pin_sources()
    pins_sha = digest(pins)
    plan = {'schema': SCHEMA, 'protocol': PROTOCOL, 'protocol_sha256': digest(PROTOCOL),
            'pins': pins, 'pins_sha256': pins_sha,
            'sources': [{k: v for k, v in item.items() if k != 'source'} for item in sources]}
    plan_path = root / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ContractError('Existing diagnostic plan differs; use a new versioned root')
    write(plan_path, plan)
    receipts = []
    for index, item in enumerate(sources):
        write(root / 'status.json', {'schema': SCHEMA, 'phase': 'running', 'stage': 'token_audit',
              'completed_sources': index, 'expected_unique_sources': len(sources),
              'source': item['key'], 'updated_utc': now()})
        receipts.append(audit_source(item, root, pins_sha))
    verify_pins(pins)
    aggregates, collisions = [], []
    for receipt in receipts:
        folder = root / 'sources' / (receipt['source_pool'] + '__' + receipt['source'])
        for name, target in [('token_information.csv', aggregates), ('observed_equalities.csv', collisions)]:
            with (folder / name).open(newline='') as stream:
                target.extend(csv.DictReader(stream))
    csv_write(export / 'token_information.csv', aggregates)
    csv_write(export / 'observed_equalities.csv', collisions)
    write(export / 'protocol.json', {'protocol': PROTOCOL, 'parent_plan_sha256': PLAN_SHA,
                                     'source_receipts': receipts, 'pins_sha256': pins_sha})
    (export / 'README.md').write_text(
        '# Bounded change/context diagnostic\n\n'
        'This label-blind diagnostic samples at most 128 original rows from each of 24 frozen source pools, '
        'retains selected source refusals, and fits no models. Byte-identical source populations are aliases. '
        'Selection uses original row identities and positions, never labels or measured performance.\n\n'
        '`token_information.csv` reports full-count cancellation and shared-context statistics. '
        '`observed_equalities.csv` separates equal full-count signatures, equal support signatures, and equal '
        '2048-bit parity sketches within each finite sample. Conditional rates use the explicitly reported '
        'different-signature pair denominators. They are not estimates of exhaustive chemical collisions.\n\n'
        'For existing hashed token IDs, D=P-R, U=P+R and S=min(R,P) obey U=abs(D)+2S. '
        'The full-count pairs (D,U) and (D,S) are reversible descriptions of the same R,P counts. '
        'Support reduction discards magnitudes; sketching and the split 1024+1024 bit budgets can discard more information. '
        'Original roles means shared local token environments, not atom retention or condition roles. '
        'Native token IDs are hashes, and neither token nor sketch equality certifies chemical identity.\n\n'
        'Rows with original eligibility or parsing refusals remain in sampled-row and refusal totals, '
        'but never contribute zero placeholders to equality measurements. Local row receipts contain no reaction strings '
        'or targets. The audit supplies descriptive evidence only, with no predictive superiority or SOTA claim.\n')
    outputs = {str(path): sha(path) for path in [plan_path, *sorted(export.glob('*'))] if path.is_file()}
    for receipt in receipts:
        path = root / 'sources' / (receipt['source_pool'] + '__' + receipt['source']) / 'COMPLETE.json'
        outputs[str(path)] = sha(path)
        outputs.update(receipt['outputs'])
    value = {'schema': SCHEMA, 'phase': 'complete', 'completed_utc': now(),
             'expected_declared_sources': sum(len(keys) for _, keys in SOURCES),
             'completed_unique_sources': len(receipts), 'sampled_rows': sum(r['sampled_rows'] for r in receipts),
             'pins': pins, 'pins_sha256': pins_sha, 'protocol_sha256': digest(PROTOCOL),
             'elapsed_seconds': time.monotonic() - started, 'outputs': outputs}
    write(root / 'COMPLETE.json', value)
    write(root / 'status.json', {key: value[key] for key in ('schema', 'phase', 'completed_utc',
          'completed_unique_sources', 'sampled_rows', 'elapsed_seconds')})
    return value


def preflight():
    initialize()
    import unittest
    test = module('ncs31_token_audit_tests', HERE / 'test_token_audit.py')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test))
    if not result.wasSuccessful():
        raise ContractError('Token diagnostic preflight failed')
    return {'schema': SCHEMA, 'preflight': 'passed', 'tests': result.testsRun}


def producer_ready(root):
    """Optional attempt-bound receipt for the parent controller, after audit."""
    stage = os.environ.get('NCS_EVIDENCE_STAGE_ID')
    key = os.environ.get('NCS_EVIDENCE_INPUT_KEY')
    if stage is None and key is None:
        return
    if not isinstance(stage, str) or re.fullmatch(r'[0-9a-fA-F]{32}', stage) is None or not key:
        raise ContractError('Invalid controller producer receipt routing')
    root = Path(root).resolve()
    result = root / 'COMPLETE.json'
    completed(root)
    write(root.parent / 'stages' / stage / 'PRODUCER_READY.json', {
        'phase': 'complete', 'stage_id': stage, 'input_key': key,
        'producer_sha256': sha(HERE / 'token_audit.py'),
        'result_path': str(result), 'result_sha256': sha(result),
        'created_utc': now(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--preflight', action='store_true')
    choice.add_argument('--run', '--once', action='store_true', dest='run')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--export', type=Path, default=EXPORT)
    args = parser.parse_args()
    try:
        value = preflight() if args.preflight else run(args.root, args.export)
        if args.run:
            producer_ready(args.root)
    except Exception as error:
        if args.run:
            write(args.root / 'status.json', {'schema': SCHEMA, 'phase': 'failed',
                  'error_type': type(error).__name__, 'updated_utc': now()})
        raise
    print(json.dumps({key: value[key] for key in value if key not in ('pins', 'outputs')}, indent=2))


if __name__ == '__main__':
    main()
