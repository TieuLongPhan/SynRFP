"""CPU-only versioned partial refresh while selected-preset clustering recovers.

The sealed final reporter remains unchanged. This adapter audits completed cells
into a separate partial directory; only its unchanged reporter can publish final
8238-cell completion after native component closure.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('sealed_final_report_for_refresh', HERE / 'report.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
ROOT = R.ROOT / 'report_refresh_v3'
EXPORT = R.EXPORT / 'report_refresh_v3'


def regression_rows():
    directory = R.EXPORT / 'regression'
    receipt = R.read_json(directory / 'COMPLETE.json')
    R.need(receipt.get('phase') == 'complete' and receipt.get('completed') == 174 and receipt.get('full_artifact_audit') is True,
           'Regression closure must precede CPU-only recovery refresh')
    R.receipt_outputs(receipt, directory)
    R.need(receipt['plan_sha256'] == R.sha(directory / 'plan.json'), 'Regression plan changed')
    rows = R.read_csv(directory / 'results.csv')
    R.need(len(rows) == 174, 'Regression grid incomplete')
    result = {'yield': [], 'barrier': []}
    for row in rows:
        R.need(row['config'] == R.SELECTED and row['method'] == R.PRIMARY and row['execution_device'] == 'cuda', 'Wrong completed regression construction')
        R.need(receipt['result_sha256'][row['result_path']] == R.sha(row['result_path']), 'Completed regression result changed')
        task = row['task']
        R.need(task in result, 'Unexpected regression task')
        result[task].append(R.add_identity(row, task, R.PRIMARY, fraction=float(row['fraction']),
             protocol=row['report_group'], population='original_common_rows', source_panel='yield13' if task == 'yield' else 'barrier14'))
    return result


def partial_cluster_rows():
    # This native module imports no torch/CUDA consumer. The process hides CUDA.
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import clustering as C
    plan = C.verify()
    rows = []
    cfg = C.load_config(C.OLD / 'config.yaml')
    dataset, ds = None, None
    for task in plan['tasks']:
        path = C.stem(task).with_suffix('.json')
        if not path.exists():
            continue
        if dataset != task['dataset']:
            dataset = task['dataset']
            ds = C.dataset_table(cfg, dataset)
        record = C.audit_cell(task, plan, ds=ds)
        R.need(record is not None, 'Existing clustering record failed audit')
        score = record['scores'][plan['panels'][dataset]['label']]
        row = dict(task='clustering', dataset=dataset, endpoint=dataset, protocol=C.PROTOCOL,
                   population=task['population'], seed=task['seed'], method=R.PRIMARY,
                   fingerprint=R.SELECTED, result_path=str(path), result_sha256=R.sha(path),
                   n_rows=record['n_rows'], n_scored=record['n_scored'], k=record['k'], **score)
        rows.append(R.add_identity(row, 'clustering', R.PRIMARY))
    binding = Path(plan['baseline_bindings'])
    R.need(C.sha(binding) == plan['pins'][str(binding.resolve())], 'Cluster baseline bindings changed')
    baselines = R.read_json(binding)['rows']
    R.need(len(baselines) == 1200, 'Incomplete cluster baseline panel')
    rows.extend(R.add_identity(row, 'clustering', row['method']) for row in baselines)
    R.check_panel(rows, ('dataset', 'population', 'seed'), 1440, complete=False)
    return rows


def build_status(rows):
    counts = {task: len(rows.get(task, [])) for task in R.EXPECTED}
    R.need(all(counts[t] <= R.EXPECTED[t] for t in counts), 'Partial refresh exceeded scientific universe')
    return dict(schema='synrfp-final-comparison-partial-refresh-v3', phase='partial', refreshed_utc=R.now(),
                tasks={t: dict(completed_cells=counts[t], expected_cells=R.EXPECTED[t], complete=counts[t] == R.EXPECTED[t]) for t in counts},
                total_completed=sum(counts.values()), total_expected=8238,
                configuration_selected=R.SELECTED, independent_confirmation=False,
                overall_complete=False, native_final_receipt_required=True, gpu_context_created=False)


def refresh():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    plan = R.verify()
    rows = R.stable_reuse({})
    for task, added in regression_rows().items():
        rows[task].extend(added)
    rows['clustering'] = partial_cluster_rows()
    R.validate_joins(rows, dict(regression=dict(phase='complete'), clustering=dict(phase='pending')))
    receipt = build_status(rows)
    outputs = {}
    for task, cells in rows.items():
        for suffix, records in (('cells', cells), ('summary', R.summarize(cells))):
            path = EXPORT / (task + '_' + suffix + '.csv')
            R.write_csv(path, records)
            outputs[str(path.resolve())] = R.sha(path)
    receipt.update(outputs_sha256=outputs, frozen_report_plan_sha256=R.sha(R.ROOT / 'report/plan.json'),
                   source_sha256=R.sha(__file__), preserved_frozen_report=True,
                   interpretation='Audited partial progress, not final completion; task metrics retain original protocol scopes.')
    R.write_json(ROOT / 'status.json', receipt)
    R.write_json(EXPORT / 'status.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument('--once', action='store_true')
    stages.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        r = refresh()
        print('Audited partial comparison: %s / 8238 cells' % r['total_completed'], flush=True)
        if args.once:
            break
        for _ in range(6):
            final = R.ROOT / 'COMPLETE.json'
            if final.exists():
                receipt = R.read_json(final)
                R.need(receipt.get('phase') == 'complete' and receipt.get('logical_cells') == 8238, 'Invalid final comparison receipt')
                R.receipt_outputs(receipt, R.ROOT)
                R.write_json(ROOT / 'WATCH_COMPLETE.json', dict(phase='complete', final_sha256=R.sha(final),
                    finished_utc=R.now(), outputs_sha256={str(final): R.sha(final)}))
                raise SystemExit(0)
            time.sleep(30)
