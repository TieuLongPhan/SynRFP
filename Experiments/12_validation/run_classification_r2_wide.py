"""Canonical radius2/2048 classification extension after the yield width test.

All14 endpoints,25 frozen grouped splits, unchanged kNN/MLP implementations.
CPU feature generation verifies byte-exact radius2/1024 baseline equivalence.
GPU1 is handed over only after the barrier core completes and no fit is active.
"""
import argparse
import copy
import fcntl
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import run_crxnfp_full as B
from synrfp_exp.config import SynRFPConfig
from synrfp_exp.features import compute_synrfp,Features

HERE=Path(__file__).resolve().parent
ROOT=(HERE/'runs/classification_r2_2048_v1').resolve()
EXPORT=HERE/'publication_metadata/classification_r2_2048_v1'
FP='synrfp-wl-r2-b2048-parity-delta'
REFERENCE_ROOT=B.ROOT
YIELD_ROOT=(HERE.parent/'13_yield/runs/r3_4096_check_v1').resolve()


def prepare():
    previous=B.verify();cfg=B.frozen_cfg();ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'plan.json').exists():return verify()
    pins={str(Path(__file__).resolve()):B.L.sha256(__file__),
          str(HERE/'run_crxnfp_full.py'):B.L.sha256(HERE/'run_crxnfp_full.py'),
          str(HERE/'crxnfp_full_worker.py'):B.L.sha256(HERE/'crxnfp_full_worker.py'),
          str(HERE.parent/'03_classification/mlp_gpu_queue.py'):B.L.sha256(HERE.parent/'03_classification/mlp_gpu_queue.py'),
          str(REFERENCE_ROOT/'config.yaml'):B.L.sha256(REFERENCE_ROOT/'config.yaml')}
    for p in (HERE.parent.parent/'synrfp').rglob('*.py'):pins[str(p)]=B.L.sha256(p)
    for name in ['features.py','models.py','metrics.py']:
        p=HERE.parent/'synrfp_exp'/name;pins[str(p)]=B.L.sha256(p)
    sources={};datasets={}
    for d in previous['classification_datasets']:
        ds=B.dataset_table(cfg,d);key=B.rxn_key(cfg,d);datasets[d]={'source':key,'rows':len(ds)}
        p=cfg['paths']['work']/'tables'/('ds_'+d+'.pkl');pins[str(p)]=B.L.sha256(p)
        for seed in range(25):
            p=cfg['paths']['work']/'splits'/d/f'cv_s{seed}.npy';pins[str(p)]=B.L.sha256(p)
        if key not in sources:
            p=cfg['paths']['work']/'tables'/('rxn_'+key+'.pkl');pins[str(p)]=B.L.sha256(p)
            baseline=cfg['paths']['work']/'features'/key/(B.SYN+'.npz');pins[str(baseline)]=B.L.sha256(baseline)
            sources[key]={'table':str(p),'baseline':str(baseline),'rows':len(B.reaction_table(cfg,key))}
    plan={'created_utc':B.now(),'fingerprint':FP,'input_policy':'n1','datasets':datasets,'sources':sources,
          'expected_knn':350,'expected_mlp':350,'grouped_splits':25,'pinned_files':pins,
          'reference_root':previous['classification_reference_root'],'models':{'knn':cfg['models']['knn'],'mlp':cfg['models']['mlp']},
          'start_after_yield':str(YIELD_ROOT/'COMPLETE.json'),'feature_execution':'8 CPU threads, CUDA hidden',
          'mlp_execution':'GPU1 after all880 barrier core fits complete and controller released safely',
          'validation':'Preserve original internal sklearn early-stopping split and initialization via prepare_arrays; no outer-test labels in worker',
          'retention':'Keep all weights/checkpoints/predictions locally; original results untouched',
          'comparison':'Exploratory after original classification factorial reviewed; wider inputs add MLP parameters'}
    B.write(ROOT/'plan.json',plan);B.write(ROOT/'plan_seal.json',{'sha256':B.L.sha256(ROOT/'plan.json')});B.write(EXPORT/'plan.json',plan)
    return plan


def verify():
    plan=json.loads((ROOT/'plan.json').read_text())
    if B.L.sha256(ROOT/'plan.json')!=json.loads((ROOT/'plan_seal.json').read_text())['sha256']:raise ValueError('Changed classification extension plan')
    for p,h in plan['pinned_files'].items():
        if B.L.sha256(p)!=h:raise ValueError('Changed classification extension source/input: '+p)
    return plan


def status(stage,**fields):
    value={'stage':stage,'updated_utc':B.now(),**fields};B.write(ROOT/(stage+'_status.json'),value);print(json.dumps(value),flush=True)


def await_yield(stage):
    while not (YIELD_ROOT/'COMPLETE.json').exists():
        status(stage,phase='queued',waiting_for='yield r3/4096 completion');time.sleep(15)


def features():
    p=verify();await_yield('features');cfg=B.frozen_cfg();fcfg=copy.deepcopy(cfg);fcfg['paths']['work']=ROOT;fcfg['n_jobs']=8
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for key,s in sorted(p['sources'].items(),key=lambda item:item[1]['rows']):
        path=ROOT/'features'/key/(FP+'.npz');receipt=path.with_suffix('.json')
        if receipt.exists():
            r=json.loads(receipt.read_text())
            if B.L.sha256(path)!=r['feature_sha256']:raise ValueError('Changed classification feature cache')
            continue
        # An incomplete cache is retained and regenerated under a fresh scope.
        scope=key if not path.exists() else key+'_retry_'+str(time.time_ns())
        status('features',phase='encoding',source=key,rows=s['rows'])
        rt=B.reaction_table(cfg,key)
        compute_synrfp(fcfg,scope,rt.rxn.tolist(),[SynRFPConfig.from_name(B.SYN),SynRFPConfig.from_name(FP)])
        baseline=ROOT/'features'/scope/(B.SYN+'.npz');native=ROOT/'features'/scope/(FP+'.npz')
        with np.load(baseline) as a,np.load(s['baseline']) as b:
            if not np.array_equal(a['X'],b['X']) or not np.array_equal(a['ok'],b['ok']):raise ValueError('Canonical baseline fingerprint mismatch: '+key)
        if scope!=key:
            if path.exists():path.rename(path.with_name(FP+'_incomplete_'+str(time.time_ns())+'.npz'))
            native.rename(path)
        B.write(receipt,{'source':key,'rows':len(rt),'feature_sha256':B.L.sha256(path),
            'input_sha256':p['pinned_files'][s['table']],'baseline_sha256':p['pinned_files'][s['baseline']],
            'baseline_bitwise_equivalent':True,'plan_sha256':B.L.sha256(ROOT/'plan.json')})
    status('features',phase='complete',sources=len(p['sources']))


def ready_feature(cfg,d,stage):
    key=B.rxn_key(cfg,d);path=ROOT/'features'/key/(FP+'.npz');receipt=path.with_suffix('.json')
    while not receipt.exists():
        state=ROOT/'features_status.json'
        if state.exists() and json.loads(state.read_text()).get('phase')=='failed':raise RuntimeError('Feature preparation failed')
        status(stage,phase='waiting_features',current_dataset=d);time.sleep(15)
    meta=json.loads(receipt.read_text())
    if B.L.sha256(path)!=meta['feature_sha256'] or meta['plan_sha256']!=B.L.sha256(ROOT/'plan.json'):raise ValueError('Changed classification feature provenance')
    with np.load(path) as z:feat=Features(z['X'],int(z['n_bits']),z['ok'])
    if feat.n_bits!=2048 or len(feat.X)!=meta['rows']:raise ValueError('Malformed classification feature dimensions')
    return feat,meta['feature_sha256']


def report():
    with (ROOT/'report.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        records=[json.loads(p.read_text()) for p in (ROOT/'classification/03_classification/results/benchmark').glob('*/*/*.json')]
        if records:
            f=pd.DataFrame(records);f.to_csv(EXPORT/'cells.csv',index=False)
            summary=f.groupby(['dataset','model']).agg(completed=('mcc','size'),mcc_mean=('mcc','mean'),mcc_std=('mcc','std')).reset_index()
            summary['expected']=25;summary['complete']=summary.completed==25;summary.to_csv(EXPORT/'summary.csv',index=False)
        if len(records)==700:B.write(ROOT/'COMPLETE.json',{'finished_utc':B.now(),'models':700})
    return len(records)


def release_completed_barrier_gpu1():
    barrier=HERE.parent/'14_barrier/runs/barrier_allfps_v1';unit='synrfp-barrier-dual-gpu1-v1.service';stable_since=None
    while subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0:
        core=[p for p in (barrier/'models').glob('*/*/*/result.json') if json.loads(p.read_text())['method']!='chemprop_published_architecture_v2_port']
        if len(core)==880:
            if stable_since is None:stable_since=time.monotonic()
            pid=int(subprocess.check_output(['systemctl','--user','show',unit,'--property=MainPID','--value'],text=True))
            children=[]
            for p in Path('/proc').glob('[0-9]*/stat'):
                try:
                    fields=p.read_text().rsplit(')',1)[1].split()
                    if int(fields[1])==pid:children.append(p.parent.name)
                except (OSError,ValueError,IndexError):pass
            if time.monotonic()-stable_since>=30 and not children:
                subprocess.run(['systemctl','--user','stop',unit],check=True)
                B.write(barrier/'gpu1_core_complete.json',{'finished_utc':B.now(),'core_models':880,'active_training_children':0,'gpu1_released_for':'classification r2/2048','all_weights_preserved':True})
                break
        else:stable_since=None
        status('gpu',phase='queued',waiting_for='barrier GPU1 core completion',barrier_core_completed=len(core));time.sleep(15)


def install():
    B.ROOT=ROOT;B.EXPORT=EXPORT;B.status=status;B.report=report;B.ready_feature=ready_feature
    B.ordered=lambda cfg:sorted(cfg['datasets'],key=lambda d:len(B.dataset_table(cfg,d)))
    # Keep all frozen numerical worker/preparation code; change only the input
    # feature and use unscaled bits, exactly as the original SynRFP benchmark.
    gpu=inspect.getsource(B.run_gpu)
    gpu=gpu.replace("        subprocess.run([GPU_PYTHON, '-u', str(HERE / 'crxnfp_full_encoder.py')], check=True, env=env)\n",'')
    gpu=gpu.replace('crxnfp',FP).replace(FP+'_full_worker.py','crxnfp_full_worker.py')
    gpu=gpu.replace("mu, sd = Xtr.mean(0), Xtr.std(0)+1e-6", "mu, sd = np.zeros(Xtr.shape[1],np.float32), np.ones(Xtr.shape[1],np.float32)")
    gpu=gpu.replace('encoded_sources=9','encoded_sources=0')
    knn=inspect.getsource(B.run_knn).replace('crxnfp',FP)
    exec(compile(gpu,str(__file__),'exec'),B.__dict__);exec(compile(knn,str(__file__),'exec'),B.__dict__)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','features','knn','gpu'],required=True);a=parser.parse_args()
    if a.stage=='prepare':prepare()
    else:
        with (ROOT/(a.stage+'.lock')).open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                verify();await_yield(a.stage)
                if a.stage=='features':features()
                else:
                    cfg=B.frozen_cfg()
                    if a.stage=='gpu':release_completed_barrier_gpu1()
                    install()
                    {'knn':B.run_knn,'gpu':B.run_gpu}[a.stage](cfg,6 if a.stage=='knn' else 4)
            except BaseException as error:
                status(a.stage,phase='failed',error=repr(error));raise
