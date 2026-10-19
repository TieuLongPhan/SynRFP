"""Finish 174 selected-preset regressors using the unchanged legacy CUDA worker.

No historical model is replaced. Only immutable fit inputs are referenced from
the CPU probes, after reconstruction from native features, rows and targets.
"""
from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = Path.home() / '.local/share/synrfp-runs/final_comparison_v1/regression'
EXPORT = HERE / 'publication_metadata/final_comparison_v1/regression'
WORKER = EXP / '13_yield/continuous_mlp.py'
PYTHON = '/home/labhhc4/anaconda3/envs/rxnfp/bin/python'
LEASE = EXP / '12_validation/runs/gpu_1.lock'
PRESET = EXP / '33_shared_preset_decision/publication_metadata/shared_preset_decision_v1/FROZEN_PRESET.json'
CONFIG = 'synrfp-wl-r2-b2048-parity-delta'
METHOD = 'synrfp_selected'
NATIVE_METHOD = 'synrfp_r2b2048'
TORCH = '2.4.0.post301'
PARAMS = dict(hidden=512, learning_rate=.001, weight_decay=.0001,
              batch_size=256, max_epochs=200, patience=15, min_delta=.0001, seed=42)
_HASHES = {}
_HASH_HITS = 0


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    global _HASH_HITS
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _HASHES:
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for part in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                h.update(part)
        after = path.stat()
        if key[1:] != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('File changed during full SHA256 read: '+str(path))
        _HASHES[key] = h.hexdigest()
    else:
        _HASH_HITS += 1
        after=path.stat()
        if key[1:] != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('File changed during immutable hash lookup: '+str(path))
    return _HASHES[key]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def status(phase, **value):
    value.update(phase=phase, updated_utc=now())
    write(ROOT / 'status.json', value)
    write(EXPORT / 'status.json', value)
    print(json.dumps(value), flush=True)


def metrics(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.ndim != 1 or truth.shape != predicted.shape or not len(truth):
        raise ValueError('Malformed held-out predictions')
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError('Nonfinite evaluation')
    denominator = float(np.square(truth-truth.mean()).sum())
    numerator = float(np.square(truth-predicted).sum())
    return dict(mae=float(np.abs(truth-predicted).mean()),
                rmse=float(np.sqrt(numerator / len(truth))),
                r2=1.-numerator/denominator if denominator else float(numerator == 0))


def selectors(split_path, common, fraction):
    with np.load(split_path, allow_pickle=False) as z:
        rows = tuple(z[k][common[z[k]]] for k in ('train', 'validation', 'test'))
    if fraction < 1:
        order = np.random.RandomState(42).permutation(len(rows[0]))
        rows = (rows[0][order[:max(1, int(len(order)*fraction))]], rows[1], rows[2])
    combined = np.concatenate(rows)
    if len(np.unique(combined)) != len(combined) or any(len(r) < 10 for r in rows):
        raise ValueError('Invalid native split populations')
    return rows


def training_arrays(feature, labels, rows):
    """Extract the exact native preprocessing function, without controller side effects."""
    path = EXP / '13_yield/run_continuous.py'
    function = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'training_arrays')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    packed = np.asarray(feature)
    value = SimpleNamespace(is_bits=True,
        dense=lambda r: np.unpackbits(packed[r], axis=1, count=2048).astype(np.float32))
    return namespace['training_arrays'](value, labels, rows)


def require_arrays(path, expected):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(expected):
            raise ValueError('Unexpected array keys: ' + str(path))
        for name, value in expected.items():
            actual = z[name]
            if actual.dtype != np.asarray(value).dtype or not np.array_equal(actual, value):
                raise ValueError('Native input mismatch: ' + str(path) + ':' + name)


def input_path(parent, result, name):
    candidates = [key for key in result['artifacts'] if Path(key).name == name]
    if len(candidates) != 1:
        raise ValueError('Missing or ambiguous immutable probe input: ' + name)
    path = parent / candidates[0]
    if sha(path) != result['artifacts'][candidates[0]]:
        raise ValueError('Changed probe input: ' + str(path))
    return path.resolve()


def raw_jobs():
    for task, base_name, probe_name in (
            ('yield', '13_yield/runs/continuous_allfps_v1', '13_yield/runs/r2_2048_check_v1'),
            ('barrier', '14_barrier/runs/barrier_allfps_v1', '14_barrier/runs/r2_2048_cpu_v1')):
        base_root, probe_root = (EXP / base_name).resolve(), (EXP / probe_name).resolve()
        base, probe = read(base_root/'plan.json'), read(probe_root/'plan.json')
        for source_root in (base_root,probe_root):
            seal=read(source_root/'plan_seal.json')
            if sha(source_root/'plan.json') != seal.get('sha256',seal.get('plan_sha256')):
                raise ValueError('Original source plan seal differs')
        if probe['base_plan_sha256'] != sha(base_root/'plan.json'):
            raise ValueError('Probe no longer binds the original benchmark')
        if probe['params'] != PARAMS or probe['expected_models'] != (64 if task=='yield' else 110):
            raise ValueError('Unexpected native probe scope/recipe')
        if task == 'yield':
            cells = [dict(spec=s, fraction=1.) for s in base['splits']]
        else:
            cells = probe['cells']
        for cell in cells:
            spec, fraction = cell['spec'], cell['fraction']
            name = spec['split'] if task == 'yield' else spec['split']+'_f'+str(int(fraction*100))
            result_path = probe_root/'models'/spec['dataset']/name/NATIVE_METHOD/'result.json'
            feature = (probe_root/'features'/spec['dataset']/(NATIVE_METHOD+'.npz')) if task == 'yield' else probe_root/'features'/(spec['dataset']+'.npz')
            yield dict(task=task, dataset=spec['dataset'], split=spec['split'], fraction=fraction,
                report_group=spec['report_group'], probe_result=str(result_path), feature=str(feature),
                split_path=spec['path'], table=base['tables'][spec['dataset']]['path'],
                coverage=str(base_root/('coverage_'+spec['dataset']+'.npz')),
                target='yield_pct' if task == 'yield' else 'barrier_kcal_mol',
                base_plan=str(base_root/'plan.json'), probe_plan=str(probe_root/'plan.json'))


def prepare():
    if (ROOT/'plan.json').exists():
        return verify()
    ROOT.mkdir(parents=True, exist_ok=True)
    preset = read(PRESET)
    if preset['configuration_selected'] != CONFIG or preset['configuration'] != dict(
            tokenizer='wl', radius=2, sketch='parity', mode='delta', n_bits=2048,
            kind='bits', seed=1, bbit=2, count_policy='native_unweighted_support_after_full_multiset_algebra'):
        raise ValueError('Unexpected selected configuration')
    pins = {}
    def pin(path):
        path = str(Path(path).resolve()); pins[path] = sha(path); return pins[path]
    for path in (Path(__file__), WORKER, PRESET, EXP/'13_yield/run_continuous.py', EXP/'synrfp_exp/features.py'):
        pin(path)
    jobs, pools = [], {}
    for raw in raw_jobs():
        source = Path(raw['probe_result']); result = read(source)
        key = (raw['task'], raw['dataset'])
        for name in ('probe_result','feature','split_path','table','coverage','base_plan','probe_plan'):
            pin(raw[name])
        for name in ('base_plan','probe_plan'):
            pin(Path(raw[name]).parent/'plan_seal.json')
        feature_metadata=Path(raw['feature']).with_suffix('.json')
        pin(feature_metadata)
        meta=read(feature_metadata)
        if meta['feature_sha256'] != sha(raw['feature']):
            raise ValueError('Changed native feature receipt')
        if raw['task']=='yield':
            identity=meta['identity']
            if (meta['variant'] != NATIVE_METHOD or identity['tokenizer']!='wl' or
                    identity['sketch']!='parity' or identity['fingerprint_seed']!=1 or
                    identity['variants'][NATIVE_METHOD] != dict(radius=2,n_bits=2048,mode='delta')):
                raise ValueError('Selected yield construction differs')
            for path,expected in identity['implementation_sha256'].items():
                if pin(path)!=expected: raise ValueError('Changed native fingerprint implementation')
        if result.get('method', result.get('fingerprint')) != NATIVE_METHOD:
            raise ValueError('Unexpected feature recipe')
        if result['feature_sha256'] != sha(raw['feature']) or result['split_sha256'] != sha(raw['split_path']):
            raise ValueError('Probe feature or split identity changed')
        if key not in pools:
            table = pd.read_csv(raw['table'])
            with np.load(raw['feature'], allow_pickle=False) as z:
                packed, ok = z['X'], z['ok']
                if int(z['n_bits']) != 2048 or packed.dtype != np.uint8 or packed.shape != (len(table),256):
                    raise ValueError('Unexpected selected fingerprint shape')
            with np.load(raw['coverage'], allow_pickle=False) as z:
                common = z['shared']
            if common.dtype != bool or common.shape != (len(table),) or not ok[common].all():
                raise ValueError('Changed original valid population')
            pools.clear()  # At most one source pool in memory.
            pools[key] = table, packed, common
        table, packed, common = pools[key]
        rows = selectors(raw['split_path'], common, raw['fraction'])
        inputs = {name: str(input_path(source.parent, result, name))
                  for name in ('fit.npz','test.npy','preprocessing.npz','rows.npz')}
        old_job_path = input_path(source.parent, result, 'job.json')
        pin(old_job_path)
        old_job = read(old_job_path)
        if old_job['params'] != PARAMS or old_job['dataset'] != raw['dataset'] or old_job['split'] != raw['split']:
            raise ValueError('Probe training recipe changed')
        with np.load(inputs['rows.npz'], allow_pickle=False) as z:
            for name, value in zip(('train','validation','test'), rows):
                if not np.array_equal(z[name], value):
                    raise ValueError('Probe row population differs from original protocol')
        fit, test, preprocessing = training_arrays(packed, table[raw['target']].to_numpy(np.float64), rows)
        require_arrays(inputs['fit.npz'], fit)
        require_arrays(inputs['preprocessing.npz'], preprocessing)
        actual = np.load(inputs['test.npy'], allow_pickle=False)
        if actual.dtype != test.dtype or not np.array_equal(actual, test):
            raise ValueError('Probe held-out feature input differs')
        for path in inputs.values(): pin(path)
        identity = raw['task']+'__'+raw['dataset']+'__'+raw['split']+'__f'+str(int(raw['fraction']*100))
        jobs.append(dict(raw, id=identity, inputs=inputs, params=PARAMS,
            source_probe_result_sha256=sha(source), split_sha256=sha(raw['split_path']),
            feature_sha256=sha(raw['feature']), ntrain=len(rows[0]), nval=len(rows[1]), ntest=len(rows[2])))
        if len(jobs) % 25 == 0:
            status('preparing', verified_inputs=len(jobs), total=174)
    if len(jobs) != 174 or len({j['id'] for j in jobs}) != 174:
        raise ValueError('Incomplete selected regression task grid')
    plan = dict(schema='synrfp-final-selected-regression-v1', created_utc=now(),
        config=CONFIG, method=METHOD, expected_models=174, jobs=jobs, pins=pins,
        params=PARAMS, worker=str(WORKER), python=PYTHON, required_torch_version=TORCH,
        execution_device='cuda', physical_gpu=1, input_oracle='exact native13 preprocessing and original masked/fraction selectors',
        population_policy='Original seven-FP shared validity; no new configuration mask',
        limitations=['Exploratory preset selected after prior outer-test results were seen; no new generalization claim.'],
        weights_local_only=True, no_historical_files_modified=True)
    write(ROOT/'plan.json', plan)
    write(ROOT/'plan_seal.json', dict(plan_sha256=sha(ROOT/'plan.json')))
    write(EXPORT/'plan.json', plan)
    status('prepared', completed=0, total=174, plan_sha256=sha(ROOT/'plan.json'))
    return plan


def verify(full=False):
    if full:
        _HASHES.clear()
    plan = read(ROOT/'plan.json')
    if sha(ROOT/'plan.json') != read(ROOT/'plan_seal.json')['plan_sha256']:
        raise ValueError('Changed final regression plan')
    if plan['expected_models'] != 174 or plan['config'] != CONFIG:
        raise ValueError('Changed final regression scope')
    for path, expected in plan['pins'].items():
        if sha(path) != expected:
            raise ValueError('Changed final regression input/source: ' + path)
    return plan


def native_env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')


REPLAY = r'''import json,sys,numpy as np,torch
from pathlib import Path
from torch import nn
p=Path(sys.argv[1]);c=torch.load(p/'best_model.pt',map_location='cpu',weights_only=False)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
m=nn.Sequential(nn.Linear(c['input_dim'],c['params']['hidden']),nn.ReLU(),nn.Linear(c['params']['hidden'],1)).cuda()
m.load_state_dict(c['state_dict']);m.eval()
x=np.load(p/'test.npy',allow_pickle=False)[:32]
with np.load(p/'preprocessing.npz',allow_pickle=False) as z: mean,scale=float(z['target_mean']),float(z['target_scale'])
with torch.no_grad(): actual=m(torch.as_tensor(x,dtype=torch.float32,device='cuda')).squeeze(-1).cpu().numpy().astype(np.float64)*scale+mean
expected=np.load(p/'predictions.npy',allow_pickle=False)[:32]
if not np.allclose(actual,expected,atol=1e-5,rtol=1e-5): raise ValueError('Saved native checkpoint replay failed')
print(json.dumps(dict(passed=True,samples=len(x),max_abs_difference=float(np.max(np.abs(actual-expected))),torch_version=torch.__version__)))
'''


def check_native(directory, job, replay=False):
    directory = Path(directory)
    if set(job) != {'dataset','split','fingerprint','params','plan_sha256','inputs'} or set(job['inputs']) != {'fit.npz','test.npy','preprocessing.npz'}:
        raise ValueError('Unexpected native worker input contract')
    for name, expected in job['inputs'].items():
        if sha(directory/name) != expected:
            raise ValueError('Changed final worker input')
    training = read(directory/'training.json')
    if training['worker_sha256'] != sha(WORKER) or training['job_sha256'] != sha(directory/'job.json'):
        raise ValueError('Native training provenance differs')
    if training['physical_gpu'] != 1 or training['torch_version'] != TORCH:
        raise ValueError('Native training numeric backend differs')
    for name, expected in training['artifacts'].items():
        if sha(directory/name) != expected:
            raise ValueError('Changed retained native weight/prediction artifact')
    history = training['history']
    if not history or not all(np.isfinite(v['validation_mse_standardized']) for v in history):
        raise ValueError('Invalid native validation history')
    chosen = min(history, key=lambda v: v['validation_mse_standardized'])
    if chosen['epoch'] != training['selected_epoch'] or chosen['validation_mse_standardized'] != training['best_validation_mse_standardized']:
        raise ValueError('Wrong selected validation checkpoint')
    if replay:
        output = subprocess.check_output([PYTHON,'-c',REPLAY,str(directory)],env=native_env(),text=True)
        value = json.loads(output.strip().splitlines()[-1])
        if not value['passed'] or value['torch_version'] != TORCH:
            raise ValueError('Native checkpoint replay did not pass')
        write(directory/'replay.json', value)
    elif not read(directory/'replay.json')['passed']:
        raise ValueError('Missing completed checkpoint replay')
    return training


def native_job(directory, task, plan_sha):
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in task['inputs'].items():
        path = directory/name
        if path.exists() or path.is_symlink():
            if not path.is_symlink() or path.resolve() != Path(source).resolve():
                raise ValueError('Changed input reference')
        else:
            path.symlink_to(source)
    value = dict(dataset=task['dataset'], split=task['split'], fingerprint=METHOD,
                 params=task['params'], plan_sha256=plan_sha,
                 inputs={n:sha(directory/n) for n in ('fit.npz','test.npy','preprocessing.npz')})
    write(directory/'job.json', value)
    return value


def audit_result(task, result):
    directory = Path(result['attempt_directory'])
    if result['plan_sha256'] != sha(ROOT/'plan.json') or result['config'] != CONFIG or result['id'] != task['id']:
        raise ValueError('Different completed final regression identity')
    for name in ('task','dataset','split','fraction','report_group','ntrain','nval','ntest',
                 'source_probe_result_sha256','split_sha256','feature_sha256'):
        if result[name] != task[name]:
            raise ValueError('Completed regression task metadata differs: '+name)
    if result['method'] != METHOD or result['native_method'] != NATIVE_METHOD or result['seed'] != 42:
        raise ValueError('Completed regression method identity differs')
    for name, expected in result['artifacts'].items():
        if sha(directory/name) != expected:
            raise ValueError('Changed completed final regression artifact')
    job = read(directory/'job.json')
    if (job['params'] != PARAMS or job['plan_sha256'] != result['plan_sha256'] or
            job['dataset'] != task['dataset'] or job['split'] != task['split'] or job['fingerprint'] != METHOD):
        raise ValueError('Changed completed worker job')
    check_native(directory,job)
    with np.load(directory/'evaluation.npz',allow_pickle=False) as z:
        values=metrics(z['observed'],z['prediction'])
        rows=np.load(task['inputs']['rows.npz'],allow_pickle=False)['test']
        if not np.array_equal(z['source_rows'],rows): raise ValueError('Changed final evaluation row order')
        truth=pd.read_csv(task['table'],usecols=[task['target']])[task['target']].to_numpy(float)[rows]
        if not np.array_equal(z['observed'],truth): raise ValueError('Changed final evaluation truth')
        for name,value in values.items():
            if not np.isclose(result[name],value,atol=1e-12,rtol=0): raise ValueError('Changed final metric')
    return result


def report(plan, audit=True):
    records=[]
    for task in plan['jobs']:
        path=ROOT/'models'/task['id']/'result.json'
        if path.exists():
            value=read(path)
            if audit: audit_result(task,value)
            records.append(value)
    if records:
        fields=[k for k in records[0] if k!='artifacts']
        EXPORT.mkdir(parents=True,exist_ok=True)
        temporary=EXPORT/'results.csv.tmp'
        with temporary.open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
        temporary.replace(EXPORT/'results.csv')
    if len(records)==174:
        receipt=dict(schema='synrfp-final-selected-regression-complete-v1',phase='complete',completed_utc=now(),
            completed=174,expected_models=174,plan_sha256=sha(ROOT/'plan.json'),
            config=CONFIG,native_worker_sha256=sha(WORKER),torch_version=TORCH,
            outputs_sha256={'results.csv':sha(EXPORT/'results.csv')},
            result_sha256={str(ROOT/'models'/t['id']/'result.json'):sha(ROOT/'models'/t['id']/'result.json') for t in plan['jobs']},
            full_artifact_audit=audit,weights_local_only=True)
        if not audit: raise ValueError('Completion requires the full artifact audit')
        write(ROOT/'COMPLETE.json',receipt);write(EXPORT/'COMPLETE.json',receipt)
    return len(records)


def preflight(cuda=False):
    """CPU mutation tests; optional tiny native CUDA/checkpoint smoke under its lease."""
    import unittest
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_regression.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise RuntimeError('Regression preflight tests failed')
    receipt=dict(passed=True,tests=result.testsRun,checked_utc=now(),cuda=False,source_sha256=sha(__file__))
    if cuda:
        with LEASE.open('r+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            directory=ROOT/'preflight'/('cuda-'+now().replace(':','-'));directory.mkdir(parents=True)
            rng=np.random.RandomState(1)
            X=rng.randint(0,2,(96,16)).astype(np.float32);y=(X[:,0]-.5).astype(np.float32)
            np.savez_compressed(directory/'fit.npz',X_train=X[:64],X_val=X[64:80],y_train=y[:64],y_val=y[64:80])
            np.save(directory/'test.npy',X[80:]);np.savez_compressed(directory/'preprocessing.npz',target_mean=np.float64(0),target_scale=np.float64(1))
            p=dict(PARAMS,hidden=8,max_epochs=8,patience=3)
            job=dict(dataset='preflight',split='synthetic',fingerprint=METHOD,params=p,plan_sha256='fixture',
                inputs={n:sha(directory/n) for n in ('fit.npz','test.npy','preprocessing.npz')})
            write(directory/'job.json',job)
            subprocess.run([PYTHON,'-u',str(WORKER),str(directory)],env=native_env(),check=True)
            check_native(directory,job,replay=True)
            receipt.update(cuda=True,replay=read(directory/'replay.json'),directory=str(directory))
    write(ROOT/'PREFLIGHT.json',receipt);write(EXPORT/'PREFLIGHT.json',receipt)
    if cuda:
        ready=dict(receipt,phase='ready')
        write(ROOT/'PILOT_READY.json',ready);write(EXPORT/'PILOT_READY.json',ready)
    return receipt


def run():
    pilot=read(ROOT/'PREFLIGHT.json')
    if not pilot.get('cuda') or not pilot.get('passed') or pilot['source_sha256'] != sha(__file__):
        raise RuntimeError('Native CUDA preflight must pass before full execution')
    plan=verify(full=True)
    with LEASE.open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for task in plan['jobs']:
            verify()
            parent=ROOT/'models'/task['id'];parent.mkdir(parents=True,exist_ok=True)
            with (parent/'claim.lock').open('a+') as claim:
                fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if (parent/'result.json').exists():
                    audit_result(task,read(parent/'result.json'));continue
                if shutil.disk_usage(ROOT).free < 40*1024**3:
                    raise RuntimeError('Below40GiB storage reserve; retain all weights and stop admission')
                number=1
                while (parent/('attempt-%03d'%number)).exists():number+=1
                directory=parent/('attempt-%03d'%number)
                job=native_job(directory,task,sha(ROOT/'plan.json'))
                status('training',task=task['id'],completed=sum((ROOT/'models'/t['id']/'result.json').exists() for t in plan['jobs']),total=174)
                with (directory/'worker.log').open('w') as out:
                    subprocess.run([PYTHON,'-u',str(WORKER),str(directory)],env=native_env(),stdout=out,stderr=subprocess.STDOUT,check=True)
                training=check_native(directory,job,replay=True)
                rows=np.load(task['inputs']['rows.npz'],allow_pickle=False)['test']
                table=pd.read_csv(task['table']);truth=table[task['target']].to_numpy(float)[rows]
                predicted=np.load(directory/'predictions.npy',allow_pickle=False)
                values=metrics(truth,predicted)
                np.savez_compressed(directory/'evaluation.npz',observed=truth,prediction=predicted,source_rows=rows,row_ids=table.row_id.to_numpy()[rows])
                value=dict(id=task['id'],task=task['task'],dataset=task['dataset'],split=task['split'],fraction=task['fraction'],
                    report_group=task['report_group'],method=METHOD,native_method=NATIVE_METHOD,config=CONFIG,seed=42,
                    ntrain=task['ntrain'],nval=task['nval'],ntest=task['ntest'],**values,
                    plan_sha256=sha(ROOT/'plan.json'),split_sha256=task['split_sha256'],feature_sha256=task['feature_sha256'],
                    source_probe_result_sha256=task['source_probe_result_sha256'],native_worker_sha256=sha(WORKER),
                    torch_version=training['torch_version'],execution_device='cuda',physical_gpu=1,seconds=training['seconds'],
                    selected_epoch=training['selected_epoch'],completed_utc=now(),attempt_directory=str(directory),result_path=str(parent/'result.json'),
                    artifacts={p.name:sha(p) for p in directory.iterdir() if p.is_file()})
                write(parent/'result.json',value);audit_result(task,value)
                count=sum((ROOT/'models'/t['id']/'result.json').exists() for t in plan['jobs'])
                if count<174: report(plan,audit=False)
                status('running' if count<174 else 'auditing',completed=count,total=174)
        verify(full=True)
        count=report(plan,audit=True)
        if count!=174: raise RuntimeError('Incomplete final selected regression grid')
        status('complete',completed=count,total=174)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--run',action='store_true')
    group.add_argument('--preflight',action='store_true');group.add_argument('--report',action='store_true')
    group.add_argument('--pilot',action='store_true')
    parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.pilot:preflight(True)
    elif args.preflight:preflight(args.cuda)
    elif args.report:report(verify(full=True))
    else:run()
