#!/usr/bin/env python
"""Full canonical crxnfp comparison, separate from the completed frozen suite.

Stages run independently under external CPU/RAM caps. GPU 1 alone is used for
official frozen embedding inference and the unchanged original MLP protocol.
All rows, 25 classification splits, 12 clustering endpoints, and ten clustering
seeds are frozen before execution. No tuning or label-hidden resolution claim.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as L
from synrfp_exp.config import expand_fingerprints, load_config
from synrfp_exp.data import dataset_table, reaction_table, rxn_key
from synrfp_exp.features import load
from synrfp_exp.metrics import classification
from synrfp_exp.models import fit_predict
from synrfp_exp.splits import TRAIN, VAL, TEST
from threadpoolctl import threadpool_limits

ROOT = (HERE / 'runs/crxnfp_full_v1').resolve()
EXPORT = HERE / 'publication_metadata/crxnfp_full_v1'
OLD = (HERE / 'runs/ncs_extensions/modern_encoder').resolve()
GPU_PYTHON = '/home/labhhc4/anaconda3/envs/rxnfp/bin/python'
SYN = 'synrfp-wl-r2-b1024-parity-delta'
CLASS_PANEL = [SYN, 'drfp', 'rdkit_struct', 'rdkit_diff', 'rxnfp', 'rxnfp_mlm']
CLUSTER_PANEL = [SYN, 'drfp', 'rdkit_struct', 'rdkit_diff', 'rxnfp_mlm']


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


H = module('full_crxnfp_cluster_helpers', HERE / '07_clustering.py')
M = module('full_crxnfp_old_modern', HERE / '10_ncs_extensions.py')
Q = module('full_crxnfp_mlp_preparation', HERE.parent / '03_classification/mlp_gpu_queue.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    H.atomic_json(path, value)


def csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(path)


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise RuntimeError(f'Wrong existing symlink: {path}')
    elif path.exists():
        raise RuntimeError(f'Refusing to replace existing file: {path}')
    else:
        path.symlink_to(target.resolve())


def frozen_cfg():
    return load_config(ROOT / 'config.yaml')


def code(cfg, ds_name, seed):
    path = cfg['paths']['work'] / 'splits' / ds_name / f'cv_s{seed}.npy'
    if not path.exists():
        raise RuntimeError(f'Matched split missing; no split regeneration allowed: {path}')
    a = np.load(path, allow_pickle=False)
    if not np.isin(a, [TRAIN, VAL, TEST]).all():
        raise RuntimeError(f'Invalid split: {path}')
    return a


def status(stage, **kwargs):
    p = ROOT / f'{stage}_status.json'
    value = json.loads(p.read_text()) if p.exists() else {'started_utc': now()}
    value.update(pid=os.getpid(), updated_utc=now(), **kwargs)
    write(p, value)
    print(json.dumps({'stage': stage, **kwargs}), flush=True)


def pin(inputs, path):
    path = path.resolve(strict=True)
    inputs[str(path)] = L.sha256(path)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'plan.json').exists():
        verify()
        print('Existing frozen plan verified; preparation unchanged', flush=True)
        return
    original = load_config(HERE / 'configs/n1.yaml')
    cfg = copy.deepcopy(yaml.safe_load((HERE / 'configs/n1.yaml').read_text()))
    for key, value in original['paths'].items():
        cfg['paths'][key] = str(value)
    cfg['paths']['work'] = str(ROOT / 'work')
    cfg['paths']['results_root'] = str(ROOT / 'classification')
    cfg['benchmark']['fingerprints'] = ['crxnfp']
    cfg['benchmark']['models'] = ['knn', 'mlp']
    cfg['rxnfp_dataset_fingerprints'] = {}
    cfg['n_jobs'] = 12
    cfg['baselines']['crxnfp'] = {'external': True}
    (ROOT / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    work = ROOT / 'work'
    for name in ('tables', 'splits', 'groups'):
        link(work / name, original['paths']['work'] / name)
    inputs = {}
    pin(inputs, HERE / 'configs/n1.yaml')
    for p in sorted((HERE.parent / 'synrfp_exp').glob('*.py')):
        pin(inputs, p)
    for p in [Path(__file__), HERE / 'crxnfp_full_encoder.py', HERE / 'crxnfp_full_worker.py',
              HERE / 'modern_encoder.py', HERE / '07_clustering.py', HERE / '10_ncs_extensions.py',
              HERE.parent / '03_classification/mlp_gpu_queue.py',
              HERE.parent / '03_classification/mlp_cuda_worker.py', ROOT / 'config.yaml']:
        pin(inputs, p)
    for p in [HERE / 'external/crxnfp_local/checkpoint-45600/model.safetensors',
              HERE / 'external/crxnfp_local/tokenization.py', HERE / 'external/crxnfp_local/smiles_util.py',
              HERE / 'runs/completion/crxnfp_provenance.json', OLD / 'protocol.json', ROOT / 'smoke_validation.json']:
        pin(inputs, p)
    datasets = list(original['datasets'])
    keys = sorted({rxn_key(original, d) for d in datasets})
    for key in keys:
        table = reaction_table(original, key)
        pin(inputs, original['paths']['work'] / 'tables' / f'rxn_{key}.pkl')
        target = ROOT / 'encoder/inputs' / f'{key}.csv.gz'
        target.parent.mkdir(parents=True, exist_ok=True)
        old_source = OLD / 'inputs' / target.name
        if old_source.exists():
            old_table = pd.read_csv(old_source)
            if not np.array_equal(old_table.rxn.astype(str), table.rxn.astype(str)):
                raise RuntimeError(f'Original modern encoder source differs: {key}')
            shutil.copyfile(old_source, target)
            for suffix in ('.npz', '.json'):
                source = OLD / 'features' / f'{key}{suffix}'
                if suffix == '.npz':
                    with np.load(source, allow_pickle=False) as z:
                        if len(z['X']) != len(table) or z['X'].shape[1] != 256:
                            raise RuntimeError(f'Original feature row/width mismatch: {key}')
                    link(ROOT / 'encoder/features' / source.name, source)
                else:
                    destination = ROOT / 'encoder/features' / source.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                pin(inputs, source)
        elif not target.exists():
            table[['rxn']].to_csv(target, index=False, compression={'method': 'gzip', 'mtime': 0})
        pin(inputs, target)
        for fp in CLASS_PANEL:
            source = original['paths']['work'] / 'features' / key / f'{fp}.npz'
            pin(inputs, source)
            link(work / 'features' / key / source.name, source)
        # A symlink to a future atomic feature publication is intentional.
        p = work / 'features' / key / 'crxnfp.npz'
        if not p.is_symlink():
            p.symlink_to(ROOT / 'encoder/features' / f'{key}.npz')
    old_class = original['paths']['results_root'] / '03_classification/results/benchmark'
    for d in datasets:
        ds = dataset_table(original, d)
        pin(inputs, original['paths']['work'] / 'tables' / f'ds_{d}.pkl')
        rt = reaction_table(original, rxn_key(original, d))
        rows = ds.row.to_numpy()
        if not np.array_equal(rt.r_id.iloc[rows].astype(str), ds.r_id.astype(str)):
            raise RuntimeError(f'Row IDs mismatch: {d}')
        for seed in range(25):
            a = code(original, d, seed)
            if len(a) != len(ds):
                raise RuntimeError(f'Split length mismatch: {d}/{seed}')
            pin(inputs, original['paths']['work'] / 'splits' / d / f'cv_s{seed}.npy')
            for model in ('knn', 'mlp'):
                for fp in CLASS_PANEL:
                    p = old_class / d / f'cv_s{seed}' / f'{fp}__{model}.json'
                    rec = json.loads(p.read_text())
                    if (rec['dataset'] != d or rec['seed'] != seed or rec['model'] != model
                            or rec['fingerprint'] != fp or rec['scheme'] != 'cv'
                            or rec['n_test'] != int((a == TEST).sum())
                            or rec['n_train'] != int((a != TEST).sum())
                            or not np.isfinite(rec['mcc'])):
                        raise RuntimeError(f'Unmatched classification reference: {p}')
                    pin(inputs, p)
    for root in [OLD / 'clustering', OLD / 'clustering_shared_valid',
                 (HERE / 'runs/clustering/n1').resolve()]:
        for p in root.rglob('*'):
            if p.suffix in ('.json', '.npz'):
                pin(inputs, p)
    plan = {
        'study': 'crxnfp_full_v1', 'created_utc': now(), 'input_policy': 'n1',
        'classification_datasets': datasets, 'classification_models': ['knn', 'mlp'],
        'classification_seeds': list(range(25)), 'classification_new_cells': 700,
        'clustering_datasets': H.SMALL + H.LARGE, 'clustering_seeds': list(range(10)),
        'clustering_all_row_cells': 120, 'clustering_shared_valid_cells': 720,
        'classification_reference_root': str(old_class),
        'clustering_reference_root': str((HERE / 'runs/clustering/n1').resolve()),
        'classifiers': original['models'], 'small_clustering': H.SMALL_PARAMS,
        'large_clustering': H.LARGE_PARAMS, 'physical_gpu': 1,
        'resolution': 'unchanged full-dataset reference class count; known-resolution benchmark',
        'classification_failure_policy': 'same all-row protocol; failed encodings retained as zero vectors and coverage reported',
        'clustering_failure_policy': 'all-row arm plus common valid-row mask for all six label-free encoders',
        'reuse': 'old four-endpoint modern fits and baseline assignments only after hash/row/mask/settings validation',
        'encoder': json.loads((HERE / 'runs/completion/crxnfp_provenance.json').read_text()),
        'resources': {'gpu_stage_cpu_threads': 4, 'knn_cpu_threads': 12, 'clustering_cpu_threads': 8,
                      'gpu_stage_ram_gib': 20, 'knn_ram_gib': 32, 'clustering_ram_gib': 32},
        'limitations': ['Repeated CV folds are dependent.', 'Frozen encoder training exposure and unknown exact overlap must be disclosed.',
                        'No tuning; no label-hidden resolution claim; original frozen study gates remain unchanged.'],
        'inputs_sha256': inputs,
    }
    write(ROOT / 'plan.json', plan)
    write(ROOT / 'plan_seal.json', {'sha256': L.sha256(ROOT / 'plan.json')})
    EXPORT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'plan.json', EXPORT / 'plan.json')
    print(json.dumps({k: v for k, v in plan.items() if k not in ('inputs_sha256', 'encoder', 'classifiers')}), flush=True)


def verify():
    plan = json.loads((ROOT / 'plan.json').read_text())
    if L.sha256(ROOT / 'plan.json') != json.loads((ROOT / 'plan_seal.json').read_text())['sha256']:
        raise RuntimeError('Frozen plan seal differs')
    for name, digest in plan['inputs_sha256'].items():
        if L.sha256(Path(name)) != digest:
            raise RuntimeError(f'Frozen input/source changed: {name}')
    return plan


def ready_feature(cfg, d, stage):
    key = rxn_key(cfg, d)
    path = ROOT / 'encoder/features' / f'{key}.npz'
    meta = path.with_suffix('.json')
    while not (path.exists() and meta.exists()):
        status(stage, phase='waiting_for_encoder', current_dataset=d)
        gp = ROOT / 'gpu_status.json'
        if gp.exists() and json.loads(gp.read_text()).get('phase') == 'failed':
            raise RuntimeError(f'GPU inference failed; feature unavailable: {key}')
        time.sleep(30)
    record = json.loads(meta.read_text())
    if record['feature_sha256'] != L.sha256(path):
        raise RuntimeError(f'Encoder feature SHA differs: {key}')
    source = ROOT / 'encoder/inputs' / f'{key}.csv.gz'
    if record['inputs']['input_sha256'] != L.sha256(source):
        raise RuntimeError(f'Encoder input SHA differs: {key}')
    f = load(cfg, key, 'crxnfp')
    rt = reaction_table(cfg, key)
    if len(f.X) != len(rt) or f.X.shape[1] != 256 or not np.isfinite(f.X).all():
        raise RuntimeError(f'Encoder feature/source mismatch: {key}')
    return f, L.sha256(path)


def complete_record(path):
    if not path.exists():
        return False
    x = json.loads(path.read_text())
    if x['plan_sha256'] != L.sha256(ROOT / 'plan.json'):
        raise RuntimeError(f'Mixed run provenance: {path}')
    for name, digest in x.get('artifacts_sha256', {}).items():
        if L.sha256(ROOT / name) != digest:
            raise RuntimeError(f'Prediction/model artifact SHA differs: {name}')
    return True


def ordered(cfg):
    # Existing four source encodings first; then small ready results, large tails.
    return sorted(cfg['datasets'], key=lambda d: (not (OLD / 'features' / f'{rxn_key(cfg,d)}.npz').exists(), len(dataset_table(cfg,d))))


def run_knn(cfg, threads):
    n = 0
    for d in ordered(cfg):
        ds = dataset_table(cfg, d)
        feats, digest = ready_feature(cfg, d, 'knn')
        y, rows = ds.y.to_numpy(), ds.row.to_numpy()
        for seed in range(25):
            out = ROOT / 'classification/03_classification/results/benchmark' / d / f'cv_s{seed}/crxnfp__knn.json'
            if complete_record(out):
                n += 1
                continue
            a = code(cfg, d, seed); tr = np.flatnonzero(a != TEST); va = np.flatnonzero(a == VAL); te = np.flatnonzero(a == TEST)
            status('knn', phase='running', completed=n, total=350, current_dataset=d, seed=seed)
            start = time.monotonic()
            with threadpool_limits(limits=threads):
                pred = fit_predict('knn', cfg['models']['knn'], feats, rows[tr], y[tr], rows[va], y[va], rows[te], seed, threads)
            artifact = out.parent / 'crxnfp__knn_predictions.npz'
            H.atomic_npz(artifact, predictions=pred, y=y[te], source_row=rows[te], r_id=ds.r_id.iloc[te].astype(str).to_numpy(dtype=str))
            write(out, {'study': 'benchmark', 'dataset': d, 'scheme': 'cv', 'seed': seed,
                        'fingerprint': 'crxnfp', 'model': 'knn', 'input_policy': 'n1',
                        'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
                        'n_unfeaturised_test': int((~feats.ok[rows[te]]).sum()), **classification(y[te], pred),
                        'elapsed_seconds': time.monotonic()-start, 'completed_utc': now(),
                        'feature_sha256': digest, 'split_sha256': L.sha256(cfg['paths']['work'] / 'splits' / d / f'cv_s{seed}.npy'),
                        'plan_sha256': L.sha256(ROOT / 'plan.json'),
                        'artifacts_sha256': {str(artifact.relative_to(ROOT)): L.sha256(artifact)}})
            n += 1
        del feats
        report()
    status('knn', phase='complete', completed=n, total=350, finished_utc=now())


def run_gpu(cfg, threads):
    status('gpu', phase='inference', completed=0, total=350, physical_gpu=1)
    with (HERE / 'runs/gpu_1.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS=str(threads),
                   OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        subprocess.run([GPU_PYTHON, '-u', str(HERE / 'crxnfp_full_encoder.py')], check=True, env=env)
        status('gpu', phase='mlp', completed=0, total=350, encoded_sources=9)
        log = (ROOT / 'mlp_worker.log').open('a', buffering=1)
        proc = subprocess.Popen([GPU_PYTHON, '-u', str(HERE / 'crxnfp_full_worker.py'), '--device', '0'],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
        try:
            response = json.loads(proc.stdout.readline())
            if not response.get('ready'):
                raise RuntimeError('MLP worker did not initialize')
            n = 0
            for d in ordered(cfg):
                ds = dataset_table(cfg, d)
                feats, digest = ready_feature(cfg, d, 'gpu')
                rows, y = ds.row.to_numpy(), ds.y.to_numpy()
                for seed in range(25):
                    out = ROOT / 'classification/03_classification/results/benchmark' / d / f'cv_s{seed}/crxnfp__mlp.json'
                    if complete_record(out):
                        n += 1
                        continue
                    a = code(cfg, d, seed)
                    outer = np.concatenate([np.flatnonzero(a == TRAIN), np.flatnonzero(a == VAL)])
                    te = np.flatnonzero(a == TEST)
                    directory = ROOT / 'models' / d / f'cv_s{seed}'
                    directory.mkdir(parents=True, exist_ok=True)
                    start = time.monotonic()
                    status('gpu', phase='mlp', completed=n, total=350, current_dataset=d, seed=seed, physical_gpu=1)
                    Xtr, Xte = feats.dense(rows[outer]), feats.dense(rows[te])
                    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-6
                    Xtr, Xte = (Xtr-mu)/sd, (Xte-mu)/sd
                    classes = Q.prepare_arrays(directory, Xtr, y[outer], Xte, cfg['models']['mlp'], seed, f'{d}/cv_s{seed}/crxnfp')
                    H.atomic_npz(directory / 'preprocessing.npz', mu=mu, sd=sd, classes=classes, outer=outer, test=te)
                    del Xtr, Xte
                    proc.stdin.write(json.dumps({'directory': str(directory)})+'\n'); proc.stdin.flush()
                    response = json.loads(proc.stdout.readline())
                    if not response.get('ok'):
                        raise RuntimeError(response.get('error','Worker failed'))
                    pred = classes[np.load(directory / 'predictions.npy')]
                    H.atomic_npz(directory / 'test_predictions.npz', predictions=pred, y=y[te], source_row=rows[te], r_id=ds.r_id.iloc[te].astype(str).to_numpy(dtype=str))
                    paths = [directory / name for name in ('best_weights.npz','preprocessing.npz','state.npz','job.json','predictions.npy','test_predictions.npz')]
                    rec = {'study':'benchmark','dataset':d,'scheme':'cv','seed':seed,'fingerprint':'crxnfp','model':'mlp',
                           'backend':'torch_cuda','protocol':'sklearn_adam_fp32_v1','input_policy':'n1','physical_gpu':1,
                           'n_train':len(outer),'n_val':int((a==VAL).sum()),'n_test':len(te),
                           'n_unfeaturised_test':int((~feats.ok[rows[te]]).sum()), **classification(y[te],pred),
                           **response['result'],'training_params':cfg['models']['mlp'], 'completed_utc':now(),
                           'elapsed_seconds':time.monotonic()-start,'feature_sha256':digest,
                           'split_sha256':L.sha256(cfg['paths']['work'] / 'splits' / d / f'cv_s{seed}.npy'),
                           'plan_sha256':L.sha256(ROOT / 'plan.json'),
                           'artifacts_sha256':{str(p.relative_to(ROOT)):L.sha256(p) for p in paths}}
                    write(out, rec)
                    # Retain every trained weight and prediction locally; only disposable input matrices go.
                    for name in ('train.npy','test.npy'):
                        (directory / name).unlink()
                    n += 1
                del feats
                report()
            status('gpu',phase='complete',completed=n,total=350,finished_utc=now())
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate(); proc.wait()
            log.close()


def checked_reuse(path, ds, used, valid, feature_sha, dataset_sha, comparison=None):
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    assignment = path.with_suffix('.npz')
    if L.sha256(assignment) != rec['assignment_sha256']:
        raise RuntimeError(f'Original assignment SHA differs: {assignment}')
    if comparison is not None:
        old = rec['comparison']
        if old['dataset_sha256'] != dataset_sha or old['source_feature_sha256'] != comparison['source_feature_sha256']:
            return None
    elif rec.get('fingerprint') == 'crxnfp':
        meta = json.loads(Path(rec['feature_metadata_file']).read_text())
        if meta['feature_sha256'] != feature_sha:
            return None
    with np.load(assignment,allow_pickle=False) as z:
        if not (np.array_equal(z['source_row'],ds.row.to_numpy()) and np.array_equal(z['y'],ds.y.to_numpy())
                and np.array_equal(z['r_id'].astype(str),ds.r_id.astype(str).to_numpy())
                and np.array_equal(z['used'],used) and np.array_equal(z['valid'],valid)):
            return None
        labels = z['labels'].copy()
    return labels, {'reused_assignment_file':str(assignment),'reused_assignment_sha256':rec['assignment_sha256'],
                    'original_record_sha256':L.sha256(path),'original_scores':rec['scores']}


def run_clustering(cfg, threads):
    from sklearn.cluster import KMeans, MiniBatchKMeans
    total = 0
    endpoints = sorted(H.SMALL+H.LARGE, key=lambda d:(not (OLD/'features'/f'{rxn_key(cfg,d)}.npz').exists(),len(dataset_table(cfg,d))))
    for d in endpoints:
        ds = dataset_table(cfg,d); rows = ds.row.to_numpy(); k=int(ds.y.nunique())
        feats,digest=ready_feature(cfg,d,'clustering')
        X=feats.dense(rows); valid=feats.ok[rows]; all_rows=np.ones(len(ds),dtype=bool)
        dsha=L.sha256(cfg['paths']['work']/'tables'/f'ds_{d}.pkl')
        params=H.SMALL_PARAMS if d in H.SMALL else H.LARGE_PARAMS
        masks={'crxnfp':valid}; hashes={'crxnfp':digest}
        for fp in CLUSTER_PANEL:
            f=load(cfg,rxn_key(cfg,d),fp); masks[fp]=f.ok[rows]
            hashes[fp]=L.sha256(cfg['paths']['work']/'features'/rxn_key(cfg,d)/f'{fp}.npz')
            del f
        shared=M.shared_encoding_mask(masks)
        if shared.sum()<k:raise RuntimeError(f'Insufficient matched rows: {d}')
        comparison={'panel':CLUSTER_PANEL+['crxnfp'],'dataset_sha256':dsha,'source_feature_sha256':hashes,
                    'n_rows':len(ds),'n_shared_valid':int(shared.sum()),'shared_valid_coverage':float(shared.mean()),
                    'invalid_encodings_by_fingerprint':{fp:int((~v).sum()) for fp,v in masks.items()}}
        write(ROOT/'clustering/coverage'/f'{d}.json',comparison)
        for row_policy,panel,used in [('all_rows',['crxnfp'],all_rows),('shared_valid_all_six',CLUSTER_PANEL+['crxnfp'],shared)]:
            for fp in panel:
                dense=None
                for seed in range(10):
                    out=ROOT/'clustering'/row_policy/d/f'{fp}__s{seed}.json'
                    if complete_record(out):total+=1;continue
                    status('clustering',phase='running',completed=total,total=840,current_dataset=d,fingerprint=fp,seed=seed,row_policy=row_policy)
                    start=time.monotonic();reuse=None
                    old=OLD/('clustering' if row_policy=='all_rows' else 'clustering_shared_valid')/d/f'{fp}__s{seed}.json'
                    reuse=checked_reuse(old,ds,used,masks[fp],hashes[fp],dsha,comparison if row_policy!='all_rows' else None)
                    if reuse is None and row_policy!='all_rows' and fp!='crxnfp':
                        reuse=M.reuse_shared_assignment(cfg,d,fp,seed,ds,used,masks[fp],hashes[fp],dsha)
                    if reuse is None and row_policy!='all_rows' and fp=='crxnfp' and shared.all():
                        original=ROOT/'clustering/all_rows'/d/f'crxnfp__s{seed}.json'
                        reuse=checked_reuse(original,ds,used,masks[fp],hashes[fp],dsha)
                    if reuse:
                        labels,info=reuse
                    else:
                        if dense is None:
                            dense=X[used] if fp=='crxnfp' else load(cfg,rxn_key(cfg,d),fp).dense(rows[used])
                        estimator=KMeans(n_clusters=k,random_state=seed,**params) if d in H.SMALL else MiniBatchKMeans(n_clusters=k,random_state=seed,**params)
                        with threadpool_limits(limits=threads):fitted=estimator.fit_predict(dense)
                        labels=np.full(len(ds),-1,dtype=np.int32);labels[used]=fitted;info={}
                    assignment=out.with_suffix('.npz')
                    H.atomic_npz(assignment,labels=labels,source_row=rows,r_id=ds.r_id.astype(str).to_numpy(dtype=str),y=ds.y.to_numpy(),used=used,valid=masks[fp])
                    scores=info.pop('original_scores',None) or H.scores(ds,cfg['datasets'][d],labels,used)
                    write(out,{'dataset':d,'fingerprint':fp,'seed':seed,'input_policy':'n1','row_policy':row_policy,
                               'method':'kmeans' if d in H.SMALL else 'minibatch_kmeans','params':params,'k':k,
                               'scores':scores,'n_rows':len(ds),'n_scored':int(used.sum()),'n_valid':int(masks[fp].sum()),
                               'valid_fraction':float(masks[fp].mean()),'comparison':comparison,
                               'feature_sha256':hashes[fp],'dataset_sha256':dsha,'completed_utc':now(),
                               'elapsed_seconds':time.monotonic()-start,'plan_sha256':L.sha256(ROOT/'plan.json'),
                               'feature_metadata_file':str(ROOT/'encoder/features'/f'{rxn_key(cfg,d)}.json') if fp=='crxnfp' else None,
                               'assignment_sha256':L.sha256(assignment),'artifacts_sha256':{str(assignment.relative_to(ROOT)):L.sha256(assignment)},**info})
                    total+=1
                del dense
        del X,feats
        report()
    status('clustering',phase='complete',completed=total,total=840,finished_utc=now())


def report():
    EXPORT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'report.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        plan=json.loads((ROOT/'plan.json').read_text())
        fresh=[]
        for p in (ROOT/'classification/03_classification/results/benchmark').rglob('crxnfp__*.json'):
            fresh.append(json.loads(p.read_text()))
        records=list(fresh)
        for d in plan['classification_datasets']:
            for seed in range(25):
                for model in ('knn','mlp'):
                    for fp in CLASS_PANEL:
                        p=Path(plan['classification_reference_root'])/d/f'cv_s{seed}'/f'{fp}__{model}.json'
                        records.append(json.loads(p.read_text()))
        df=pd.DataFrame(records)
        cols=['dataset','fingerprint','model','seed','mcc','accuracy','macro_f1','weighted_f1','balanced_accuracy','n_train','n_test','n_unfeaturised_test']
        csv(EXPORT/'classification_cells.csv',df[cols])
        df['repeat']=df.seed//5
        rep=df.groupby(['dataset','fingerprint','model','repeat']).mcc.agg(['mean','count']).reset_index()
        complete=rep[rep['count']==5]
        summary=complete.groupby(['dataset','fingerprint','model'])['mean'].agg(['mean','std','count']).reset_index()
        summary=summary.rename(columns={'mean':'mcc_mean','std':'mcc_std_across_repeat_means','count':'complete_repeats'})
        csv(EXPORT/'classification_summary.csv',summary)
        syn=df[df.fingerprint==SYN][['dataset','model','seed','mcc']].rename(columns={'mcc':'synrfp_mcc'})
        crx=df[df.fingerprint=='crxnfp'][['dataset','model','seed','mcc']].rename(columns={'mcc':'crxnfp_mcc'})
        paired=syn.merge(crx,on=['dataset','model','seed']);paired['synrfp_minus_crxnfp']=paired.synrfp_mcc-paired.crxnfp_mcc
        csv(EXPORT/'classification_paired.csv',paired)
        clusters=[]
        cfg=frozen_cfg()
        for p in (ROOT/'clustering').rglob('*__s*.json'):
            x=json.loads(p.read_text())
            ds=x['dataset'];label=cfg['datasets'][ds]['label']
            clusters.append({'dataset':ds,'fingerprint':x['fingerprint'],'seed':x['seed'],'row_policy':x['row_policy'],
                             'n_rows':x['n_rows'],'n_scored':x['n_scored'],'valid_fraction':x['valid_fraction'],**x['scores'][label],
                             'reused':bool(x.get('reused_assignment_file'))})
        csv(EXPORT/'clustering_cells.csv',clusters)
        baseline_clusters=[]
        for d in plan['clustering_datasets']:
            method='kmeans' if d in H.SMALL else 'minibatch_kmeans'
            for fp in CLUSTER_PANEL:
                for seed in range(10):
                    path=Path(plan['clustering_reference_root'])/'all_rows'/d/fp/f'{method}__s{seed}.json'
                    x=json.loads(path.read_text())
                    baseline_clusters.append({'dataset':d,'fingerprint':fp,'seed':seed,'row_policy':'all_rows',
                                              'n_rows':x['n_rows'],'n_scored':x['n_scored'],'valid_fraction':x['valid_fraction'],
                                              **x['scores'][cfg['datasets'][d]['label']],'reused':True})
        csv(EXPORT/'clustering_all_rows_comparison.csv',baseline_clusters+[x for x in clusters if x['row_policy']=='all_rows'])
        if clusters:
            c=pd.DataFrame(clusters+baseline_clusters)
            summ=c.groupby(['dataset','fingerprint','row_policy']).agg(ami_mean=('ami','mean'),ami_std=('ami','std'),ari_mean=('ari','mean'),ari_std=('ari','std'),seeds=('seed','count')).reset_index()
            csv(EXPORT/'clustering_summary.csv',summ)
        counts={'classification_knn':sum(x['model']=='knn' for x in fresh),'classification_mlp':sum(x['model']=='mlp' for x in fresh),
                'clustering_all_rows':sum(x['row_policy']=='all_rows' for x in clusters),'clustering_shared_valid':sum(x['row_policy']=='shared_valid_all_six' for x in clusters)}
        expected={'classification_knn':350,'classification_mlp':350,'clustering_all_rows':120,'clustering_shared_valid':720}
        phases={s:json.loads((ROOT/f'{s}_status.json').read_text()).get('phase') if (ROOT/f'{s}_status.json').exists() else 'not_started' for s in ('gpu','knn','clustering')}
        done=counts==expected and all(p=='complete' for p in phases.values())
        if done:
            class_ids={(x['dataset'],x['model'],x['seed']) for x in fresh}
            if class_ids!={(d,m,s) for d in plan['classification_datasets'] for m in ('knn','mlp') for s in range(25)}:
                raise RuntimeError('Final classification cell identities incomplete')
            cluster_ids={(x['dataset'],x['fingerprint'],x['seed'],x['row_policy']) for x in clusters}
            expected_ids={(d,'crxnfp',s,'all_rows') for d in plan['clustering_datasets'] for s in range(10)}
            expected_ids|={(d,fp,s,'shared_valid_all_six') for d in plan['clustering_datasets'] for fp in CLUSTER_PANEL+['crxnfp'] for s in range(10)}
            if cluster_ids!=expected_ids:raise RuntimeError('Final clustering cell identities incomplete')
            for p in (ROOT/'classification/03_classification/results/benchmark').rglob('crxnfp__*.json'):
                if not complete_record(p):raise RuntimeError(f'Incomplete final result: {p}')
            for p in (ROOT/'clustering').rglob('*__s*.json'):
                if not complete_record(p):raise RuntimeError(f'Incomplete final clustering result: {p}')
            write(ROOT/'COMPLETE.json',{'finished_utc':now(),'counts':counts,'all_artifact_hashes_verified':True})
        write(EXPORT/'status.json',{'phase':'complete' if done else 'in_progress','updated_utc':now(),'counts':counts,'expected':expected,'stages':phases,
                                  'plan_sha256':L.sha256(ROOT/'plan.json'),'weights_exported':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['prepare','gpu','knn','clustering','report','status'])
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage=='status':
        for p in ROOT.glob('*_status.json'):print(p.name,p.read_text())
        return
    verify()
    if args.stage=='report':report();return
    with (ROOT/f'{args.stage}.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status(args.stage,phase='starting')
            cfg=frozen_cfg()
            {'gpu':run_gpu,'knn':run_knn,'clustering':run_clustering}[args.stage](cfg,args.threads)
            report()
        except BaseException:
            status(args.stage,phase='failed',error=traceback.format_exc())
            raise


if __name__=='__main__':
    main()
