#!/usr/bin/env python
"""Publish only the audited seven-task fingerprint tables, using a private index.

The specialist-reference queue is deliberately outside this publication gate.
No model, raw input, cache or unrelated working-tree/index edit is staged.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = Path.home() / '.local/share/synrfp-runs/final_comparison_v1'
ROOT = RUN / 'publication'
EXPORT = HERE / 'publication_metadata/final_comparison_v1'
BRANCH = 'exp/reproducible-benchmark'
REMOTE = 'origin'
REMOTE_URL = 'git@github.com:TieuLongPhan/SynRFP.git'
SELECTED = 'synrfp-wl-r2-b2048-parity-delta'
EXPECTED = dict(classification=4200, clustering=1440, yield_=384, barrier=1944,
                feasibility=30, reagent=60, condition=180)
EXPECTED['yield'] = EXPECTED.pop('yield_')
TOTAL = 8238
SOURCE_NAMES = (
    'PROTOCOL.md', 'clustering.py', 'clustering_parallel_v2.py', 'clustering_recovery_v3.py',
    'controller.py', 'controller_parallel_v2.py', 'controller_recovery_v3.py', 'gpu0_assist.py',
    'prepare_all.py', 'regression.py', 'report.py', 'report_refresh_v3.py',
    'test_clustering.py', 'test_clustering_parallel_v2.py', 'test_clustering_recovery_v3.py',
    'test_controller.py', 'test_controller_parallel_v2.py', 'test_controller_recovery_v3.py',
    'test_regression.py', 'test_report.py', 'publish_main_results.py', 'test_publish_main_results.py',
    'PUBLICATION_PROTOCOL.md', 'synrfp-main-results-publish-v1.service',
)
SELECTION_PREFIX = 'Experiments/33_shared_preset_decision/publication_metadata/shared_preset_decision_v1/'
SELECTION_NAMES = ('FROZEN_PRESET.json', 'decision.json', 'COMPLETE.json', 'report.md',
                   'benchmark_followup.json', 'selected_endpoint_metrics.csv')
FORBIDDEN_PARTS = {'raw', 'weights', 'models', 'checkpoints', 'cache', 'caches', 'runs', '__pycache__'}
ALLOWED_SUFFIXES = {'.py', '.md', '.json', '.csv', '.service'}


def now():
    return datetime.now(timezone.utc).isoformat()


def need(test, reason):
    if not test:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def index_sha(path):
    return sha(path) if path.exists() else None


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def git(repo, *args, index=None, input=None):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0',
               GIT_SSH_COMMAND='ssh -oBatchMode=yes -oConnectTimeout=15')
    if index is not None:
        env['GIT_INDEX_FILE'] = str(index)
    result = subprocess.run(['git', '-C', str(repo), *args], env=env,
                            input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if result.returncode:
        raise RuntimeError('Git command failed (' + args[0] + '): ' + result.stderr.decode(errors='replace').strip())
    return result.stdout.decode()


def branch(repo):
    return git(repo, 'symbolic-ref', '--quiet', '--short', 'HEAD').strip()


def safe_path(repo, relative):
    relative = Path(relative)
    need(not relative.is_absolute() and '..' not in relative.parts, 'Nonrelative publication path')
    need(not set(relative.parts) & FORBIDDEN_PARTS, 'Forbidden heavyweight publication path')
    need(relative.suffix in ALLOWED_SUFFIXES, 'Forbidden publication extension')
    need('35_specialist_references' not in relative.parts, 'Unfinished specialist references cannot be published')
    path = repo / relative
    need(path.is_file() and not path.is_symlink(), 'Missing or symlinked publication file: ' + str(relative))
    need(path.resolve().is_relative_to(repo.resolve()), 'Publication path escapes checkout')
    need(path.stat().st_size <= 20*1024*1024, 'Unexpectedly large publication file')
    return path


def allowed_paths(repo=REPO):
    prefix = 'Experiments/34_final_comparison/'
    output = prefix + 'publication_metadata/final_comparison_v1/'
    paths = [prefix + name for name in SOURCE_NAMES]
    paths.extend(SELECTION_PREFIX + name for name in SELECTION_NAMES)
    paths.extend(output + task + '_' + suffix + '.csv' for task in EXPECTED for suffix in ('cells', 'summary'))
    paths.extend(output + name for name in ('COMPLETE.json', 'report.md', 'report_plan.json',
                                           'reference_inventory.json', 'PUBLICATION_SCOPE.json'))
    paths.extend(output + 'regression/' + name for name in ('COMPLETE.json', 'plan.json', 'results.csv'))
    paths.extend(output + 'clustering/' + name for name in ('final_component.json', 'plan.json',
                                                          'cells.csv', 'baseline_cells.csv', 'summary.csv'))
    return sorted(paths)


def load_report():
    spec = importlib.util.spec_from_file_location('main_results_canonical_report', HERE / 'report.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def check_header(receipt):
    need(receipt.get('phase') == 'complete', 'Canonical final comparison is not complete')
    need(receipt.get('configuration_selected') == SELECTED, 'Wrong final shared preset')
    need(receipt.get('total_expected') == TOTAL and receipt.get('logical_cells') == TOTAL
         and receipt.get('total_completed') == TOTAL, 'Final scientific cell count mismatch')
    need(set(receipt.get('tasks', {})) == set(EXPECTED), 'Seven-task receipt scope changed')
    for task, expected in EXPECTED.items():
        need(receipt['tasks'][task] == dict(complete=True, completed_cells=expected, expected_cells=expected),
             'Incomplete final task: ' + task)
    need(receipt.get('selection_interpretation') == 'exploratory_after_prior_test_results_viewed'
         and receipt.get('independent_confirmation') is False, 'Selection interpretation changed')
    need(set(receipt.get('components', {})) == {'clustering', 'regression'}
         and all(v.get('phase') == 'complete' for v in receipt['components'].values()),
         'Native component closure missing')


def full_gate():
    """Recheck native metadata/source seals and keyed six-method grids, not counts."""
    paths = [RUN/'COMPLETE.json', RUN/'report/COMPLETE.json', EXPORT/'COMPLETE.json']
    need(all(p.is_file() for p in paths), 'Canonical completion receipts are absent')
    need(len({sha(p) for p in paths}) == 1, 'Canonical final receipts disagree')
    receipt = read(paths[0])
    check_header(receipt)
    R = load_report()
    plan = R.verify(RUN)  # Every frozen canonical reuse/source/split digest.
    need(receipt['plan_sha256'] == sha(RUN/'report/plan.json'), 'Final report plan changed')
    need(sha(EXPORT/'report_plan.json') == receipt['plan_sha256'], 'Exported report plan changed')
    need(receipt['selected_preset_sha256'] == sha(R.FROZEN), 'Frozen preset changed')
    names = {task+'_'+suffix+'.csv' for task in EXPECTED for suffix in ('cells', 'summary')}
    names |= {'report.md', 'reference_inventory.json'}
    need(set(receipt['outputs_sha256']) == names, 'Final output catalog scope changed')
    R.receipt_outputs(receipt, EXPORT)
    added, states = R.components(RUN)  # Native plan/receipt/canonical result SHA and CUDA/full-audit guards.
    need(states == receipt['components'], 'Final receipt component bindings changed')
    rows = {task: R.read_csv(EXPORT/(task+'_cells.csv')) for task in EXPECTED}
    R.validate_joins(rows, states)
    for task, values in rows.items():
        need(len(values) == EXPECTED[task], 'Final output coverage mismatch')
        need(all(r['task'] == task and (r['method'] != R.PRIMARY or r['configuration'] == SELECTED)
                 for r in values), 'Final task/configuration changed')
        # A valid digest must also match the freshly reconstructed native component rows.
        if task in ('yield', 'barrier', 'clustering'):
            actual = [r for r in values if r['method'] == R.PRIMARY and
                      (task != 'barrier' or r.get('source_panel') == 'barrier14')]
            expected = added[task]
            if task == 'clustering':
                expected = [r for r in expected if r['method'] == R.PRIMARY]
            need({r['result_path']: r['result_sha256'] for r in actual} ==
                 {r['result_path']: r['result_sha256'] for r in expected}, 'Native final rows differ')
    proofs = {}
    for name in ('execution.json', 'recovery_v3/execution.json'):
        path = RUN/name
        value = read(path)
        for source, digest in value['sources_sha256'].items():
            need(sha(source) == digest, 'Executed scientific source changed: ' + source)
        proofs[name] = sha(path)
    for component, filename in (('regression', 'COMPLETE.json'), ('clustering', 'COMPLETE.json')):
        local = RUN/component/filename
        value = read(local)
        need(value.get('phase') == 'complete', 'Missing native final completion')
        need(value.get('completed') == (174 if component == 'regression' else 240), 'Native count mismatch')
        if component == 'regression':
            need(value.get('full_artifact_audit') is True, 'Regression full artifact audit missing')
        else:
            need(value.get('all_artifact_hashes_verified') is True and value.get('full_metric_replay') is True,
                 'Clustering artifact/metric replay closure missing')
        proofs[component+'/'+filename] = sha(local)
    return dict(receipt_sha256=sha(paths[0]), counts=EXPECTED, logical_cells=TOTAL,
                configuration_selected=SELECTED, scientific_proofs_sha256=proofs,
                report_plan_sha256=sha(RUN/'report/plan.json'), source_files_sha256=plan['source_files_sha256'])


def verify_setup(config):
    need(branch(REPO) == BRANCH, 'Authorized branch is no longer checked out')
    need(git(REPO, 'remote', 'get-url', '--push', REMOTE).strip() == REMOTE_URL, 'Authorized push remote changed')
    for relative, digest in config['publisher_sources_sha256'].items():
        need(sha(safe_path(REPO, relative)) == digest, 'Publisher source changed after review')


def prepare():
    need(branch(REPO) == BRANCH, 'Unexpected publication branch')
    need(git(REPO, 'remote', 'get-url', '--push', REMOTE).strip() == REMOTE_URL, 'Unexpected publication remote')
    source_paths = ['Experiments/34_final_comparison/'+name for name in SOURCE_NAMES]
    pins = {p: sha(safe_path(REPO, p)) for p in source_paths}
    value = dict(schema='synrfp-main-seven-task-publication-v1', created_utc=now(), branch=BRANCH,
                 remote=REMOTE, remote_url=REMOTE_URL, initial_head=git(REPO,'rev-parse','HEAD').strip(),
                 publication_paths=allowed_paths(), publisher_sources_sha256=pins,
                 authorization='Explicit user instruction on 2026-10-19: commit and push when 7/7 finish; do not wait for specialist references.',
                 weights_local_only=True, private_index=True, force_push=False, specialist_references_excluded=True)
    path = ROOT/'config.json'
    if path.exists():
        old = read(path)
        need(old['publication_paths'] == value['publication_paths'] and old['publisher_sources_sha256'] == pins,
             'Existing publication setup differs; retain it and version a changed publisher')
        verify_setup(old)
        return old
    write(path, value)
    write(ROOT/'REVIEW.json', dict(phase='ready_for_root_review', config_sha256=sha(path),
          available_paths=[p for p in value['publication_paths'] if (REPO/p).is_file()],
          awaiting_paths=[p for p in value['publication_paths'] if not (REPO/p).is_file()],
          real_index_sha256=index_sha(Path(git(REPO,'rev-parse','--git-path','index').strip())),
          current_staged_paths=git(REPO,'diff','--cached','--name-only').splitlines(), created_utc=now()))
    return value


@contextmanager
def index_lock(index):
    lock = index.with_name(index.name+'.lock')
    descriptor = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    identity = os.fstat(descriptor)
    try:
        yield
    finally:
        os.close(descriptor)
        if lock.exists() and (lock.stat().st_dev, lock.stat().st_ino) == (identity.st_dev, identity.st_ino):
            lock.unlink()


def commit_snapshot(repo, paths, hashes, directory, expected_branch=BRANCH):
    """CAS the branch and adopt only published index entries, preserving other stages."""
    repo, directory = Path(repo), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = git(repo, 'rev-parse', 'HEAD').strip()
    need(branch(repo) == expected_branch, 'Branch changed before staging')
    private = directory/'private.index'
    private.unlink(missing_ok=True)
    git(repo, 'read-tree', base, index=private)
    git(repo, 'add', '--', *paths, index=private)
    staged = set(git(repo, 'diff', '--cached', '--name-only', base, index=private).splitlines())
    need(staged <= set(paths), 'Private index includes unauthorized changes')
    for p, digest in hashes.items():
        need(sha(safe_path(repo, p)) == digest, 'Publication file changed while staging')
        blob = subprocess.run(['git','-C',str(repo),'show',':'+p], env=dict(os.environ,GIT_INDEX_FILE=str(private)),
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
        need(hashlib.sha256(blob).hexdigest() == digest, 'Staged bytes differ from approved snapshot')
    need(staged, 'No new final result files to commit')
    tree = git(repo, 'write-tree', index=private).strip()
    message = ('Publish final seven-task SynRFP fingerprint comparison\n\n'
               'Add 8,238 audited cells for one shared WL/r2/2048/parity/delta preset and five fingerprint baselines. '
               'Keep protocol-specific summaries, exploratory-selection limits and completion provenance. '
               'Weights, raw inputs, caches and unfinished specialist references remain local.\n')
    commit = git(repo, 'commit-tree', tree, '-p', base, input=message.encode()).strip()
    index = Path(git(repo, 'rev-parse', '--git-path', 'index').strip())
    if not index.is_absolute():
        index = repo/index
    adopt = directory/'adopt.index'
    with index_lock(index):
        need(branch(repo) == expected_branch and git(repo,'rev-parse','HEAD').strip() == base,
             'HEAD changed concurrently; publication was not committed')
        before = index_sha(index)
        overlap = git(repo,'diff','--cached','--name-only',base,'--',*paths).splitlines()
        need(not overlap, 'User staging overlaps publication paths; preserve it for inspection')
        if index.exists():
            adopt.write_bytes(index.read_bytes())
        else:
            adopt.unlink(missing_ok=True)
            git(repo,'read-tree',base,index=adopt)
        entries = git(repo,'ls-files','--stage','-z','--',*paths,index=private)
        git(repo,'update-index','-z','--index-info',index=adopt,input=entries.encode())
        for p, digest in hashes.items():
            need(sha(safe_path(repo,p)) == digest, 'Publication changed before commit')
        pending = dict(phase='prepared_ref_transaction', base=base, commit=commit,
                       branch=expected_branch, tree=tree, files_sha256=hashes,
                       index_before_sha256=before, index_after_sha256=sha(adopt), created_utc=now())
        write(directory/'TRANSACTION.json',pending)
        need(index_sha(index) == before, 'Index changed concurrently')
        git(repo,'update-ref','refs/heads/'+expected_branch,commit,base)
        os.replace(adopt,index)
        pending.update(phase='committed', committed_utc=now())
        write(directory/'COMMITTED.json',pending)
    return pending


def remote_head(repo=None):
    repo = REPO if repo is None else repo
    return git(repo,'ls-remote','--exit-code',REMOTE,'refs/heads/'+BRANCH).split()[0]


def push_committed(committed, config):
    verify_setup(config)
    commit = committed['commit']
    changes = set(git(REPO,'diff-tree','--no-commit-id','--name-only','-r',committed['base'],commit).splitlines())
    need(changes <= set(committed['files_sha256']), 'Commit includes an unapproved file')
    for relative, digest in committed['files_sha256'].items():
        result = subprocess.run(['git','-C',str(REPO),'show',commit+':'+relative],
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        need(hashlib.sha256(result.stdout).hexdigest() == digest, 'Committed publication bytes changed')
    need(git(REPO,'rev-parse','HEAD').strip() == commit, 'HEAD moved after final-results commit; will not push unrelated work')
    remote = remote_head()
    if remote != commit:
        # A normal push can advance only the approved branch without rewriting it.
        git(REPO,'push',REMOTE,commit+':refs/heads/'+BRANCH)
    need(remote_head() == commit, 'Remote did not confirm final-results commit')
    value = dict(schema='synrfp-main-results-published-v1',phase='complete',published_utc=now(),
                 commit=commit,branch=BRANCH,remote=REMOTE,remote_head=commit,
                 completed_tasks=7,logical_cells=TOTAL,task_counts=EXPECTED,
                 files_sha256=committed['files_sha256'],gate=read(ROOT/'GATE.json'),
                 weights_published=False,specialist_references_waited_for=False)
    write(ROOT/'PUBLISHED.json',value)
    line = '\nFinal main seven-task comparison published: '+commit+'; 8,238 audited cells; '+now()+'. Weights remain local; specialist references did not delay publication.\n'
    log = REPO/'SS_LOG.md'
    if log.exists() and commit not in log.read_text():
        with log.open('a') as f:
            f.write(line)
    return value


def once():
    config = read(ROOT/'config.json')
    verify_setup(config)
    if (ROOT/'PUBLISHED.json').exists():
        return read(ROOT/'PUBLISHED.json')
    if (ROOT/'COMMITTED.json').exists():
        return push_committed(read(ROOT/'COMMITTED.json'),config)
    if (ROOT/'TRANSACTION.json').exists():
        tx = read(ROOT/'TRANSACTION.json')
        need(git(REPO,'rev-parse','HEAD').strip() != tx['commit'],
             'Interrupted index transaction requires inspection; no user index is overwritten')
    if not (RUN/'COMPLETE.json').is_file():
        value = dict(phase='waiting_for_audited_7_of_7',updated_utc=now(),
                     required_logical_cells=TOTAL,specialist_references_waited_for=False,gpu_jobs=0)
        write(ROOT/'status.json',value)
        return value
    proof = full_gate()
    write(ROOT/'GATE.json',proof)
    remote = remote_head()
    # Unknown remote commits are never force pushed or automatically rebased.
    git(REPO,'merge-base','--is-ancestor',remote,'HEAD')
    write(EXPORT/'PUBLICATION_SCOPE.json',dict(schema='synrfp-published-main-tables-v1',
          configuration_selected=SELECTED, logical_cells=TOTAL, task_counts=EXPECTED,
          final_receipt_sha256=proof['receipt_sha256'],scientific_proofs_sha256=proof['scientific_proofs_sha256'],
          selection_interpretation='exploratory_after_prior_test_results_viewed',independent_confirmation=False,
          scope='Seven-task fingerprint results and scientific metadata. This is not a standalone archive of every historical training input.',
          omitted='Weights, raw datasets, caches, unrelated work and specialist-reference experiment35.',
          specialist_references_required=False))
    paths = config['publication_paths']
    hashes = {p:sha(safe_path(REPO,p)) for p in paths}
    write(ROOT/'SNAPSHOT.json',dict(files_sha256=hashes,created_utc=now(),logical_cells=TOTAL))
    # Recheck the full scientific gate immediately before the commit transaction.
    need(full_gate() == proof, 'Scientific completion changed during publication snapshot')
    committed = commit_snapshot(REPO,paths,hashes,ROOT,BRANCH)
    return push_committed(committed,config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--watch',action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'publisher.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare:
            result = prepare()
            print(json.dumps(dict(phase='prepared',paths=len(result['publication_paths']))))
            return
        need(args.once or args.watch,'Choose --prepare, --once or --watch')
        while True:
            try:
                result = once()
                write(ROOT/'status.json',result)
                print(json.dumps({k:result[k] for k in ('phase','commit','logical_cells') if k in result}),flush=True)
                if result['phase'] == 'complete' or args.once:
                    return
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                # Network failures after a safe local commit retry without making another commit.
                write(ROOT/'PUSH_PENDING.json',dict(phase='retryable_git_failure',reason=str(error),updated_utc=now()))
                if args.once:
                    raise
            except Exception as error:
                write(ROOT/'FAILED.json',dict(phase='blocked',reason=str(error),updated_utc=now()))
                raise
            time.sleep(60)


if __name__ == '__main__':
    main()
