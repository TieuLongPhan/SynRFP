"""Resolve archived baseline artifact namespaces without changing report math."""
from __future__ import annotations

import argparse
from pathlib import Path

import report_parallel_v3 as V

U, R = V.U, V.R
ROOT = V.ROOT
HERE = Path(__file__).resolve().parent
ORIGIN_ROOTS = (
    Path('/var/tmp/synrfp-labhhc4-storage/runs/crxnfp_full_v1'),
    Path('/var/tmp/synrfp-labhhc4-storage/runs/clustering'),
    R.MAIN,
)


def source_pins():
    return {str((HERE / name).resolve()): U.sha(HERE / name)
            for name in ('report_recovery_v4.py', 'test_report_recovery_v4.py')}


def resolve_artifact(native, name, roots=ORIGIN_ROOTS):
    """A declared run-relative path and a native-relative path must be unique."""
    native, name = Path(native).resolve(), Path(name)
    roots = tuple(Path(p).resolve() for p in roots)
    U.need('..' not in name.parts, 'Parent traversal in archived artifact reference')
    U.need(any(native.is_relative_to(p) for p in roots), 'Unrecognized pinned native origin')
    if name.is_absolute():
        target = name.resolve()
        U.need(any(target.is_relative_to(p) for p in roots), 'Absolute artifact escapes declared origins')
        U.need(target.is_file(), 'Missing absolute archived artifact')
        return target
    candidates = {native.parent / name}
    candidates.update(p / name for p in roots if native.is_relative_to(p))
    existing = set()
    for candidate in candidates:
        target = candidate.resolve()
        U.need(any(target.is_relative_to(p) for p in roots), 'Artifact symlink escapes declared origins')
        if target.is_file():
            existing.add(target)
    U.need(len(existing) == 1, 'Missing or ambiguous archived artifact namespace')
    return existing.pop()


def original_rows(cache, hashes, main=R.MAIN, export=R.MAIN_EXPORT, roots=ORIGIN_ROOTS):
    """Keep native row values; only repair the declared artifact path reader."""
    receipt = Path(main) / 'report/COMPLETE.json'
    complete = U.read(receipt)
    U.need(complete['phase'] == 'complete' and complete['tasks']['clustering']['complete'] is True
           and complete['tasks']['clustering']['completed_cells'] == 1440,
           'Original main clustering comparison is incomplete')
    plan = Path(main) / 'report/plan.json'
    cache.verify(plan, complete['plan_sha256'])
    path = Path(export) / 'clustering_cells.csv'
    cache.verify(path, complete['outputs_sha256'][path.name])
    for p in (receipt, plan, path):
        V.bind(p, U.sha(p), hashes, cache)
    rows = R.csv_rows(path)
    for row in rows:
        native = Path(row.get('source_result_path') or row.get('result_path', ''))
        digest = row.get('source_result_sha256') or row.get('result_sha256')
        U.need(native.is_file() and bool(digest), 'Original row lacks an authoritative native result')
        V.bind(native, digest, hashes, cache)
        saved = U.read(native)
        for name, artifact_hash in saved.get('artifacts_sha256', {}).items():
            artifact = resolve_artifact(native, name, roots)
            V.bind(artifact, artifact_hash, hashes, cache)
        if row.get('source_assignment_path'):
            assignment = resolve_artifact(native, row['source_assignment_path'], roots)
            V.bind(assignment, row['source_assignment_sha256'], hashes, cache)
    return rows


def recovery_inputs(root=ROOT):
    root = Path(root)
    paths = [root / rel for rel in (
        'clustering/plan.json', 'clustering/plan_seal.json',
        'parallel_v3/plan.json', 'parallel_v3/plan_seal.json',
        'parallel_v3/ROOT_REVIEW.json', 'control_parallel_v3/execution.json',
        'control_parallel_v3/FAILED.json',
        'control_parallel_v3/managed/attempts/preset44-v3-report.json')]
    paths += [R.MAIN / 'report/COMPLETE.json', R.MAIN / 'report/plan.json',
              R.MAIN_EXPORT / 'clustering_cells.csv']
    return {str(p.resolve()): U.sha(p) for p in paths}


def code_ready(root=ROOT, tests=0):
    U.need(type(tests) is int and tests > 0, 'Passing recovery controls required')
    return U.immutable(Path(root) / 'recovery_v4/CODE_READY.json', dict(
        schema='task-preset44-report-recovery-code-ready-v4', phase='complete',
        tests_passed=tests, sources_sha256=source_pins(), inputs_sha256=recovery_inputs(root),
        origin_roots=[str(p) for p in ORIGIN_ROOTS], scientific_report_math_unchanged=True))


def reviewed(root=ROOT, cache=None):
    root = Path(root)
    cache = cache or U.HashCache()
    ready_path, review_path = (root / 'recovery_v4' / name for name in ('CODE_READY.json', 'ROOT_REVIEW.json'))
    ready, review = U.read(ready_path), U.read(review_path)
    U.need(ready.get('schema') == 'task-preset44-report-recovery-code-ready-v4'
           and ready.get('phase') == 'complete' and ready.get('tests_passed', 0) > 0
           and ready.get('sources_sha256') == source_pins()
           and ready.get('inputs_sha256') == recovery_inputs(root)
           and ready.get('origin_roots') == [str(p) for p in ORIGIN_ROOTS],
           'Recovery readiness/source/input identity changed')
    required = {**U.read(root / 'control_parallel_v3/execution.json')['sources_sha256'], **source_pins()}
    U.need(review.get('schema') == 'task-preset44-report-recovery-root-reviewed-v4'
           and review.get('phase') == 'reviewed'
           and review.get('code_ready_sha256') == U.sha(ready_path)
           and review.get('scientific_plan_sha256') == U.sha(root / 'clustering/plan.json')
           and review.get('partition_plan_sha256') == U.sha(root / 'parallel_v3/plan.json')
           and review.get('operation') == 'resolve_pinned_baseline_artifacts_only'
           and review.get('cpu') == 2 and review.get('memory_gib') == 8 and review.get('gpu_jobs') == 0
           and review.get('scientific_report_math_unchanged') is True
           and all(review.get('sources_sha256', {}).get(p) == h for p, h in required.items()),
           'Unreviewed or changed report recovery')
    hashes = {**ready['inputs_sha256'], **review['sources_sha256']}
    hashes.update({str(p.resolve()): U.sha(p) for p in (ready_path, review_path)})
    U.verify_catalog(hashes, cache)
    return hashes


def once(root=ROOT):
    root = Path(root)
    provenance = reviewed(root)
    prior = V.ORIGINAL_ROWS
    def recovered(cache, hashes, *args, **kwargs):
        rows = original_rows(cache, hashes, *args, **kwargs)
        for p, digest in provenance.items():
            V.bind(p, digest, hashes, cache)
        return rows
    V.ORIGINAL_ROWS = recovered
    try:
        result = V.once(root)
    finally:
        V.ORIGINAL_ROWS = prior
    if result.get('phase') == 'complete':
        native = root / 'report/clustering/COMPLETE.json'
        V.bind(native, U.sha(native), {}, U.HashCache())
        receipt = dict(schema='task-preset44-report-repair-complete-v4', phase='complete',
                       candidate_cells=240, baseline_cells=1200, original_shared_control_cells=240,
                       scientific_report_math_unchanged=True, native_JSONs_modified=False,
                       sources_sha256=source_pins(), outputs_sha256={
                           str(native.resolve()): U.sha(native),
                           str((root / 'parallel_v3/REPORT_COMPLETE.json').resolve()): U.sha(root / 'parallel_v3/REPORT_COMPLETE.json'),
                           **provenance}, completed_utc=U.now())
        U.write(root / 'recovery_v4/REPAIR_COMPLETE.json', receipt)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--code-ready', action='store_true')
    parser.add_argument('--tests-passed', type=int, default=0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    print(code_ready(args.root, args.tests_passed) if args.code_ready else once(args.root))
