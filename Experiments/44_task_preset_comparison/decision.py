"""Freeze one exploratory preset per task after the corrected full factorial.

The six source families have equal influence. Selecting from already viewed
outer-fold outcomes is explicitly exploratory, and does not optimize kNN.
No chemistry or training framework is imported by this policy.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics

try:
    import common as U
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location('preset44_common_decision', Path(__file__).with_name('common.py'))
    U = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(U)

HERE, ROOT = U.HERE, U.ROOT
ROOT43 = Path.home() / '.local/share/synrfp-runs/morgan_ecfp_correction_v1'
OLD = Path.home() / '.local/share/synrfp-runs/all_endpoint_factorial_full_v1'
EXPORT43 = U.REPO / 'Experiments/43_morgan_ecfp_correction/publication_metadata/morgan_ecfp_correction_v1'
POLICY = 'six-source-family-balanced-v1'
DECLARED_POLICY = 'user_declared_path_challenger_20261020'
DECLARED_CONFIG = 'synrfp-path-r2-b2048-parity-delta'
PROFILE = 'synrfp-ecfp-count-v1'
TASKS = ('classification', 'clustering')
FAMILIES = ('schneider', 'tpl', 'uspto50k', 'syntemp', 'ecreact', 'mech')
DATASETS = {
    'classification': ('schneider_ub', 'schneider_b', 'tpl_ub', 'tpl_b', 'uspto50k_ub', 'uspto50k_b',
                       'syntemp_r0', 'syntemp_r1', 'syntemp_r2', 'ecreact_1', 'ecreact_2', 'ecreact_3', 'mech_1', 'mech_2'),
    'clustering': ('schneider_ub', 'schneider_b', 'uspto50k_ub', 'uspto50k_b', 'syntemp_r0', 'mech_1',
                   'mech_2', 'tpl_ub', 'tpl_b', 'ecreact_1', 'ecreact_2', 'ecreact_3'),
}
PRIMARY = {'classification': 'mcc', 'clustering': 'ami'}


def source_pins():
    return {str((HERE / name).resolve()): U.sha(HERE / name)
            for name in ('common.py', 'decision.py', 'test_decision.py')}


def source_family(dataset):
    matches = [name for name in FAMILIES if dataset.startswith(name + '_')]
    U.need(len(matches) == 1, 'Unknown or ambiguous source family: ' + str(dataset))
    return matches[0]


def _portable_outputs(receipt, directory, cache):
    hashes = receipt.get('outputs_sha256')
    U.need(isinstance(hashes, dict) and bool(hashes), 'Missing portable corrected43 outputs')
    for name, digest in hashes.items():
        path = Path(name)
        U.need(not path.is_absolute() and len(path.parts) == 1 and path.name not in ('.', '..'),
               'Nonportable corrected43 output name')
        cache.verify(directory / name, digest)
    U.need({'cells.csv', 'protocol.json', 'inputs.json'}.issubset(hashes), 'Missing corrected43 canonical outputs')
    return {str((directory / name).resolve()): digest for name, digest in hashes.items()}


def _validate_configs(plan, old):
    configs = old.get('configs', {})
    U.need(len(configs) == 72, 'Corrected43 reference does not define all72 configurations')
    crossing = set()
    for key, value in configs.items():
        factor = tuple(value.get(k) for k in ('tokenizer', 'radius', 'sketch', 'mode'))
        U.need(value.get('n_bits') == 2048 and value.get('kind') == 'bits' and value.get('seed') == 1,
               'Unexpected corrected43 construction specification')
        crossing.add(factor)
    wanted = set(itertools.product(('wl', 'path', 'morgan'), (2, 3), ('parity', 'minhash', 'cw'),
                                   ('delta', 'union', 'delta_union', 'roles')))
    U.need(crossing == wanted, 'Incomplete or foreign72-construction crossing')
    morgan = {key: value for key, value in configs.items() if value['tokenizer'] == 'morgan'}
    U.need(plan.get('configs') == morgan, 'Corrected Morgan specifications differ from declared72 crossing')
    return configs


def validate_cells(rows, task, plan, configs):
    U.need(task in TASKS, 'Unsupported task for preset selection')
    datasets = DATASETS[task]
    U.need(set(plan.get(task + '_datasets', [])) == set(datasets), 'Wrong corrected43 full endpoint population')
    repeat_key = 'fold' if task == 'classification' else 'seed'
    wanted = set(itertools.product(datasets, configs, range(5)))
    values, seen = {}, set()
    for row in rows:
        try:
            repeat = int(row[repeat_key])
            U.need(str(repeat) == str(row[repeat_key]).strip(), 'Nonintegral corrected43 repeat identity')
            key = row['dataset'], row['config'], repeat
            value = float(row[PRIMARY[task]])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Malformed corrected43 factorial cell') from error
        U.need(key in wanted and key not in seen and math.isfinite(value), 'Duplicate, foreign or undefined corrected43 cell')
        seen.add(key)
        spec = configs[row['config']]
        fresh = spec['tokenizer'] == 'morgan'
        U.need(row.get('result_origin') == ('fresh-study43-ECFP' if fresh else 'reused-study22-WL-Path'),
               'Old Morgan or undeclared result origin entered selection')
        U.need(row.get('tokenizer_profile') == (PROFILE if fresh else 'study22-original-wl-path'),
               'Wrong construction profile entered selection')
        U.need(all(str(row.get(k)) == str(spec[k]) for k in ('tokenizer', 'radius', 'sketch', 'mode')),
               'Corrected43 factors disagree with exact configuration specification')
        values[key] = value
    U.need(seen == wanted, 'Incomplete corrected43 keyed72-grid; no preset may be selected')
    return values


def corrected_report(task, root43=ROOT43, export43=EXPORT43, oldroot=OLD, cache=None):
    """Verify portable tables and the native source/artifact closure before ranking."""
    U.need(task in TASKS, 'Unknown selection task')
    root43, directory, oldroot = Path(root43), Path(export43) / task, Path(oldroot)
    native_path, portable_path = root43 / 'report' / task / 'COMPLETE.json', directory / 'COMPLETE.json'
    if not native_path.exists() or not portable_path.exists():
        return None
    cache = cache or U.HashCache()
    native, portable = U.read(native_path), U.read(portable_path)
    plan_path, old_path = root43 / 'plan.json', oldroot / 'plan.json'
    plan, old = U.read(plan_path), U.read(old_path)
    plan_hash = U.sha(plan_path)
    cache.verify(old_path, plan['source_plan_sha256'])
    U.need(plan.get('tokenizer_profile') == PROFILE and plan.get('row_cap') is None and plan.get('new_filtering') is False,
           'Corrected43 profile or full-row boundary differs')
    configs = _validate_configs(plan, old)
    n = len(DATASETS[task]) * 72 * 5
    for receipt in (native, portable):
        U.need(receipt.get('schema') == 'morgan43-report-status-v1' and receipt.get('phase') == 'complete'
               and receipt.get('task') == task and receipt.get('plan_sha256') == plan_hash
               and receipt.get('cells') == n and receipt.get('all_keyed_cells_verified') is True
               and receipt.get('all_retained_artifact_hashes_verified') is True
               and receipt.get('source_and_feature_closure_verified') is True
               and receipt.get('interaction_orders') == [1, 2, 3, 4], 'Corrected43 task is not fully audited')
    U.need({k: v for k, v in native.items() if k != 'outputs_sha256'} ==
           {k: v for k, v in portable.items() if k != 'outputs_sha256'}, 'Native/portable corrected43 receipts disagree')
    portable_hashes = _portable_outputs(portable, directory, cache)
    U.verify_catalog(native.get('outputs_sha256'), cache)
    U.need(all(native['outputs_sha256'].get(path) == digest for path, digest in portable_hashes.items()),
           'Portable corrected43 tables are absent from native closure')
    inputs = U.read(directory / 'inputs.json')
    U.verify_catalog(inputs, cache)
    U.need(all(native['outputs_sha256'].get(path) == digest for path, digest in inputs.items()),
           'Native corrected43 closure omits a portable provenance input')
    U.need(inputs.get(str(plan_path)) == plan_hash, 'Portable corrected43 inputs do not bind source plan')
    for catalog in (plan.get('sources_sha256'), plan.get('pinned_files')):
        U.verify_catalog(catalog, cache)
        U.need(all(inputs.get(path) == digest for path, digest in catalog.items()), 'Corrected43 report omits source/feature input')
    protocol = U.read(directory / 'protocol.json')
    U.need(protocol.get('schema') == 'morgan43-corrected72-analysis-v1' and protocol.get('task') == task
           and protocol.get('total_configs') == 72 and protocol.get('corrected_morgan_configs') == 24
           and protocol.get('reused_wl_path_configs') == 48 and protocol.get('tokenizer_profile') == PROFILE
           and protocol.get('plan_sha256') == plan_hash and protocol.get('repeats') == 5
           and protocol.get('row_cap') is None and protocol.get('orders') == [1, 2, 3, 4]
           and protocol.get('configuration_selected') is None, 'Wrong corrected43 analysis contract')
    with (directory / 'cells.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    values = validate_cells(rows, task, plan, configs)
    bindings = {**portable_hashes, str(native_path): U.sha(native_path), str(portable_path): U.sha(portable_path),
                str(plan_path): plan_hash, str(old_path): U.sha(old_path)}
    return dict(task=task, values=values, configs=configs, plan=plan, plan_sha256=plan_hash,
                report_complete_sha256=U.sha(native_path), inputs_sha256=bindings)


def average_ranks(values):
    """Descending midranks, independent of configuration insertion order."""
    groups = {}
    for config, value in values.items():
        groups.setdefault(value, []).append(config)
    ranks, position = {}, 1
    for value in sorted(groups, reverse=True):
        group = groups[value]
        rank = (position + position + len(group) - 1) / 2
        ranks.update({config: rank for config in group})
        position += len(group)
    return ranks


def rank(values, task, configs):
    """Endpoint averages, then equal family averages; never raw-row pooling."""
    panel_means = {(dataset, config): statistics.fmean(values[dataset, config, repeat] for repeat in range(5))
                   for dataset in DATASETS[task] for config in configs}
    family_values = {}
    for family in FAMILIES:
        datasets = [dataset for dataset in DATASETS[task] if source_family(dataset) == family]
        U.need(bool(datasets), 'Missing source family in corrected43 selection')
        family_values[family] = {config: statistics.fmean(panel_means[dataset, config] for dataset in datasets)
                                 for config in configs}
    ranks = {family: average_ranks(scores) for family, scores in family_values.items()}
    regrets = {}
    for family, scores in family_values.items():
        low, high = min(scores.values()), max(scores.values())
        regrets[family] = {config: (high - value) / (high - low) if high > low else 0.0
                           for config, value in scores.items()}
    output = []
    for config in configs:
        row = dict(config=config, family_balanced_mean=statistics.fmean(family_values[f][config] for f in FAMILIES),
                   mean_family_rank=statistics.fmean(ranks[f][config] for f in FAMILIES),
                   mean_normalized_regret=statistics.fmean(regrets[f][config] for f in FAMILIES),
                   worst_family_rank=max(ranks[f][config] for f in FAMILIES))
        row.update({family + '_mean': family_values[family][config] for family in FAMILIES})
        row.update({family + '_rank': ranks[family][config] for family in FAMILIES})
        output.append(row)
    output.sort(key=lambda row: (-row['family_balanced_mean'], row['mean_family_rank'],
                                row['mean_normalized_regret'], row['worst_family_rank'], row['config']))
    return [dict(selection_order=i + 1, **row) for i, row in enumerate(output)]


def verify_decision(path, task=None, cache=None):
    path, cache = Path(path), cache or U.HashCache()
    value = U.read(path)
    U.need(value.get('schema') == 'task-preset44-decision-v1' and value.get('phase') == 'frozen'
           and value.get('policy') in (POLICY, DECLARED_POLICY) and value.get('task') in TASKS
           and (task is None or value['task'] == task), 'Invalid frozen study44 decision')
    U.need(value.get('sources_sha256') == source_pins(), 'Frozen selection source changed')
    U.verify_catalog(value['sources_sha256'], cache)
    U.verify_catalog(value.get('inputs_sha256'), cache)
    U.verify_catalog(value.get('outputs_sha256'), cache)
    U.need(value.get('exploratory_selection') is True and value.get('external_confirmation') is False
           and value.get('primary_metric') == PRIMARY[value['task']]
           and value.get('shared_original_preset_retained') is True, 'Frozen selection interpretation changed')
    expected_profile = PROFILE if value['configuration']['tokenizer'] == 'morgan' else 'study22-original-wl-path'
    U.need(value.get('tokenizer_profile') == expected_profile, 'Frozen tokenizer profile differs')
    if value['policy'] == DECLARED_POLICY:
        U.need(value['task'] == 'clustering' and value.get('config') == DECLARED_CONFIG
               and value['configuration'].get('tokenizer') == 'path'
               and value.get('candidate_status') == 'provisional_user_requested_challenger'
               and value.get('automated_winner_selection') is False
               and value.get('corrected43_complete_required') is False,
               'User-declared exception is restricted to the unchanged Path challenger')
        plan_path = Path(value['source_plan_path'])
        cache.verify(plan_path, value['source_plan_sha256'])
        old = U.read(plan_path)
        U.need(old['configs'][DECLARED_CONFIG] == value['configuration'], 'Declared Path specification changed')
        U.need(value.get('plan43_sha256') is None and value.get('report43_complete_sha256') is None,
               'Declared challenger falsely claims a completed corrected factorial')
    else:
        U.need(value.get('source_families') == list(FAMILIES), 'Frozen family weights changed')
        plan_path = Path(value['source_plan_path'])
        cache.verify(plan_path, value['plan43_sha256'])
        native = Path(value['report43_complete_path'])
        cache.verify(native, value['report43_complete_sha256'])
    seal = path.with_name(path.stem + '_seal.json')
    U.need(seal.exists(), 'Frozen decision lacks independent hash seal')
    cache.verify(path, U.read(seal)['sha256'])
    return value


def declared_path(root=ROOT, oldroot=OLD, cache=None):
    """Only the explicitly requested unchanged Path preset may bypass43 closure.

    This freezes a challenger, not a factorial winner. The completed shared-WL
    benchmark remains the primary historical comparison, and corrected Morgan
    remains the execution priority. No incomplete metric grid is consulted.
    """
    root, oldroot, cache = Path(root), Path(oldroot), cache or U.HashCache()
    path = root / 'decisions/clustering_path_challenger.json'
    if path.exists():
        return verify_decision(path, 'clustering', cache)
    plan_path, seal_path = oldroot / 'plan.json', oldroot / 'plan_seal.json'
    old = U.read(plan_path)
    cache.verify(plan_path, U.read(seal_path)['sha256'])
    U.need(set(old['clustering_datasets']) == set(DATASETS['clustering']), 'Wrong immutable Path endpoint population')
    config = old['configs'][DECLARED_CONFIG]
    U.need(config == dict(tokenizer='path', radius=2, sketch='parity', mode='delta', n_bits=2048,
                          kind='bits', seed=1, bbit=2,
                          count_policy='native_unweighted_support_after_full_multiset_algebra'),
           'User-declared Path construction does not match historical specification')
    inputs = {str(plan_path): U.sha(plan_path), str(seal_path): U.sha(seal_path)}
    for dataset in DATASETS['clustering']:
        panel = old['panels'][dataset]
        table = str(Path(panel['table']).resolve())
        U.need(table in old['pinned_files'], 'Unpinned original clustering endpoint')
        inputs[table] = old['pinned_files'][table]
        source = old['sources'][panel['source']]
        receipt_path = Path(source['feature_receipt'])
        receipt = U.read(receipt_path)
        U.need(receipt.get('phase') == 'complete' and receipt.get('plan_sha256') == U.sha(plan_path),
               'Immutable Path feature cache incomplete or bound to another plan')
        inputs[str(receipt_path)] = U.sha(receipt_path)
        feature = Path(source['features'][DECLARED_CONFIG])
        for selected in (feature, feature.with_suffix('.json')):
            U.need(selected.name in receipt['files'], 'Original Path feature lacks native hash binding')
            inputs[str(selected)] = receipt['files'][selected.name]
    for raw, digest in old['pinned_files'].items():
        p = Path(raw)
        if p.suffix == '.py':
            inputs[raw] = digest
    U.verify_catalog(inputs, cache)
    directory = root / 'decisions/clustering_path_challenger_ranking'
    directory.mkdir(parents=True, exist_ok=True)
    provenance = directory / 'declaration.json'
    U.immutable(provenance, dict(config=DECLARED_CONFIG, user_requested=True,
                                automatic_metric_selection=False, matched_initialization_seeds=list(range(10)),
                                no_cross_validation_folds=True, corrected_morgan_priority=True))
    value = dict(schema='task-preset44-decision-v1', phase='frozen', task='clustering', policy=DECLARED_POLICY,
                 config=DECLARED_CONFIG, configuration=config, tokenizer_profile='study22-original-wl-path',
                 primary_metric='ami', source_families=list(FAMILIES),
                 source_plan_path=str(plan_path), source_plan_sha256=U.sha(plan_path),
                 plan43_sha256=None, report43_complete_sha256=None, report43_complete_path=None,
                 sources_sha256=source_pins(), inputs_sha256=inputs,
                 outputs_sha256={str(provenance): U.sha(provenance)},
                 shared_original_preset_retained=True, exploratory_selection=True, external_confirmation=False,
                 candidate_status='provisional_user_requested_challenger', automated_winner_selection=False,
                 auto_winner=False,
                 corrected43_complete_required=False, matched_initialization_seeds=list(range(10)),
                 frozen_utc=U.now())
    U.immutable(path, value)
    U.immutable(path.with_name(path.stem + '_seal.json'), dict(sha256=U.sha(path)))
    return value


def freeze(task, root=ROOT, root43=ROOT43, export43=EXPORT43, oldroot=OLD, cache=None):
    path, cache = Path(root) / 'decisions' / (task + '.json'), cache or U.HashCache()
    if path.exists():
        return verify_decision(path, task, cache)
    checked = corrected_report(task, root43, export43, oldroot, cache)
    if checked is None:
        return None
    ranking = rank(checked['values'], task, checked['configs'])
    directory = path.parent / (task + '_ranking')
    directory.mkdir(parents=True, exist_ok=True)
    table = directory / 'ranking.csv'
    with table.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)
    winner = ranking[0]['config']
    value = dict(schema='task-preset44-decision-v1', phase='frozen', task=task, policy=POLICY,
                 config=winner, configuration=checked['configs'][winner],
                 tokenizer_profile=PROFILE if checked['configs'][winner]['tokenizer'] == 'morgan' else 'study22-original-wl-path',
                 primary_metric=PRIMARY[task], source_families=list(FAMILIES), score=ranking[0],
                 plan43_sha256=checked['plan_sha256'], source_plan_path=str(Path(root43) / 'plan.json'),
                 report43_complete_sha256=checked['report_complete_sha256'],
                 report43_complete_path=str(Path(root43) / 'report' / task / 'COMPLETE.json'),
                 sources_sha256=source_pins(), inputs_sha256=checked['inputs_sha256'],
                 outputs_sha256={str(table.resolve()): U.sha(table)},
                 shared_original_preset_retained=True, exploratory_selection=True, external_confirmation=False,
                 mlp_selection_does_not_establish_knn_optimum=True, frozen_utc=U.now())
    U.immutable(path, value)
    U.immutable(path.with_name(path.stem + '_seal.json'), dict(sha256=U.sha(path)))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--declared-path', action='store_true')
    args = parser.parse_args()
    print(json.dumps(declared_path(args.root) if args.declared_path else freeze(args.task, args.root)))
