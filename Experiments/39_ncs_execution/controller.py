"""Persist the reviewed37v2 utility fitting then scoring pipeline on CPU only.

This controller reads fixed scientific receipts, owns two named CPU services,
and never imports a fitting framework or touches a GPU/lease/scientific input.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path.home()/'.local/share/synrfp-runs/ncs_execution_v1'
EXPORT=HERE/'publication_metadata/ncs_execution_v1'
s=importlib.util.spec_from_file_location('reviewed_utility_frozen_controller',REPO/'Experiments/26_ncs_continuous/controller.py')
C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
B=C.B
C.ROOT=ROOT/'managed';C.EXPORT=EXPORT/'managed';B.ROOT=C.ROOT;B.EXPORT=C.EXPORT
D=C.Digests()
RAW_SNAPSHOT=B.snapshot
PRODUCER=REPO/'Experiments/37_ncs_clustering_utility'
NATIVE=Path.home()/'.local/share/synrfp-runs/ncs_clustering_utility_v1/v2'
NATIVE_EXPORT=PRODUCER/'publication_metadata/ncs_clustering_utility_v2'
ENTRY=PRODUCER/'study_v2.py'
REVIEW=NATIVE/'ROOT_REVIEW.json'
ENDPOINTS=('uspto50k_b','ecreact_3')
FP='synrfp-wl-r2-b2048-parity-delta'
METHODS=(FP,'drfp','rdkit_struct','rdkit_diff','rxnfp_mlm','crxnfp')
SELECTORS=METHODS+('uniform_random',)
EVALUATORS=('drfp','rxnfp_mlm')
BUDGETS=(100,200,400,1000)
SEEDS=tuple(range(5))
OWN=('controller.py','test_controller.py','PROTOCOL.md','synrfp-ncs-execution-v1.service')


def need(value,message):
    if not value:raise C.IntegrityError(message)


def source_hashes():
    paths=[HERE/n for n in OWN]+[REPO/'Experiments/26_ncs_continuous/controller.py',REPO/'Experiments/25_followup_scheduler/budget.py']
    return {str(p.resolve()):B.sha(p) for p in paths}


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'controller_seal.json';pins=source_hashes()
    if path.exists():
        value=C.read(path);need(value['sources_sha256']==pins,'Controller source seal changed')
        return value
    value=dict(schema='synrfp-reviewed-cpu-utility-controller-v1',sources_sha256=pins,
               created_utc=B.now(),gpu_jobs=0,gpu0_reserved=True,clock_stop=False,
               controller_cpu=.1,controller_memory_bytes=256*1024**2)
    B.write(path,value);return value


def checked_pins(pins):
    need(isinstance(pins,dict) and bool(pins),'Missing source/hash bindings')
    for path,digest in pins.items():
        need(Path(path).is_absolute(),'Receipt needs absolute hash bindings')
        D.verify(path,digest)


def seal():
    path=ROOT/'controller_seal.json';need(path.exists(),'Controller must be prepared and reviewed before installation')
    own=C.read(path);checked_pins(own['sources_sha256'])
    ready_path=NATIVE/'CODE_READY.json';pilot_path=NATIVE/'pilot/PILOT_READY.json'
    if not ready_path.exists() or not pilot_path.exists() or not REVIEW.exists():return None
    ready=C.read(ready_path)
    if ready.get('phase') not in ('ready','complete','passed'):return None
    need(type(ready.get('tests_passed')) is int and ready['tests_passed']>0,'Producer needs meaningful tested readiness')
    producer_pins=ready.get('sources_sha256',{})
    need(str(ENTRY) in producer_pins and len(producer_pins)>=2,'Wrong or incomplete37v2 entrypoint binding')
    checked_pins(producer_pins)
    review=C.read(REVIEW)
    need(review.get('schema')=='synrfp37-root-reviewed-v1' and review.get('full_scope_approved') is True,
         'Root review does not authorize the full utility scope')
    need(review.get('sources_sha256')==producer_pins,'Root review is not bound to exact tested37v2 sources')
    need(review.get('pilot_receipt_sha256')==B.sha(pilot_path),'Root review lacks the exact37v2 pilot binding')
    pilot=C.read(pilot_path)
    need(pilot.get('phase')=='complete' and pilot.get('passed') is True and pilot.get('pilot') is True,
         '37v2 bounded pilot is not complete')
    need(pilot.get('selections')==28 and pilot.get('model_replays')==56 and pilot.get('all_full_predictions_replayed') is True,
         'Pilot lacks56full prediction replays and28selections')
    need(pilot.get('sources_sha256')==producer_pins,'Pilot/tested source bindings differ')
    for catalog in ('selection_receipts_sha256','model_receipts_sha256'):
        checked_pins(pilot.get(catalog,{}))
    pins=dict(own['sources_sha256']);pins.update(producer_pins)
    for p in (path,ready_path,pilot_path,REVIEW):pins[str(p.resolve())]=B.sha(p)
    manifest_path=ROOT/'execution.json'
    if manifest_path.exists():
        D.verify(manifest_path,C.read(ROOT/'execution_seal.json')['sha256'])
        value=C.read(manifest_path);checked_pins(value['sources_sha256'])
        need(value['sources_sha256']==pins,'Reviewed execution source population changed')
        return value
    value=dict(schema='synrfp-reviewed-utility-execution-v1',created_utc=B.now(),
        sources_sha256=pins,producer_sources_sha256=producer_pins,root_review_sha256=B.sha(REVIEW),
        pilot_sha256=B.sha(pilot_path),expected_selections=280,expected_models=560,
        stages=['fit','score'],gpu_jobs=0,source_study='37v2',stereo_and_study38_not_admitted=True)
    B.write(manifest_path,value);B.write(ROOT/'execution_seal.json',dict(sha256=B.sha(manifest_path)))
    return value


def resource_snapshot():
    """Retry only a proven unit-exit race; a still-active invalid cgroup blocks."""
    for attempt in range(3):
        try:return RAW_SNAPSHOT()
        except (FileNotFoundError,RuntimeError) as error:
            text=str(error)
            match=re.search(r'(synrfp[^/\s:]*?\.service)',text)
            if match is None:match=re.search(r'(?:service|quota):\s*(synrfp[^\s]+)',text)
            if match is None:raise
            unit=match.group(1)
            state=B.state(unit)
            if state in C.ACTIVE:raise C.IntegrityError('Resource audit invalid for active research unit: '+unit) from error
            B.write(ROOT/'resource_transitions'/('%d-%d.json'%(time.time_ns(),attempt)),
                    dict(unit=unit,current_state=state,error=text,attempt=attempt+1,updated_utc=B.now()))
            if attempt==2:raise C.IntegrityError('Bounded resource snapshot transition retries exhausted') from error
            time.sleep(.05)
    raise AssertionError('Unreachable resource audit state')


B.snapshot=resource_snapshot


def jobs(manifest):
    result={}
    for stage,flag,ram in [('fit','--run-full',16),('score','--score-full',8)]:
        args=[str(ENTRY),flag]
        if stage=='fit':args+=['--review',str(REVIEW)]
        result[stage]=dict(id='utility37v2-'+stage,unit='synrfp-ncs-utility37v2-'+stage+'-v1',
            python='/home/labhhc4/anaconda3/envs/synrfp/bin/python',arguments=args,
            cpu_equivalents=2,memory_gib=ram,cores='0-1',
            sources_sha256=manifest['sources_sha256'],
            environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                             OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'),
            work_done=dict(kind='receipt',path=str(ROOT/'verified'/(stage+'.json')),
                           equals=dict(phase='complete',stage=stage)))
    result['fit']['exclusive_units']=[result['score']['unit']]
    result['score']['exclusive_units']=[result['fit']['unit']]
    return result


def selection_grid():return set(itertools.product(ENDPOINTS,SELECTORS,BUDGETS,SEEDS))


def model_grid():return {(*x,e) for x in selection_grid() for e in EVALUATORS}


def normalized_catalog(catalog,expected):
    need(isinstance(catalog,dict),'Native receipt lacks its keyed catalog')
    result={}
    for path,digest in catalog.items():
        need(Path(path).is_absolute(),'Catalog paths must be absolute')
        key=str(Path(path).resolve());need(key not in result,'Duplicate canonical catalog path');result[key]=digest
    need(set(result)=={str(p.resolve()) for p in expected},'Native catalog key population differs')
    checked_pins(result);return result


def native_fit_ready(manifest):
    root=NATIVE/'full';path=root/'FITS_COMPLETE.json'
    if not path.exists():return False
    receipt=C.read(path)
    if receipt.get('phase') in ('pending','running','in_progress'):return False
    need(receipt.get('phase')=='complete' and receipt.get('schema')=='synrfp37-utility-v2-fits-complete' and receipt.get('pilot') is False,'Full native fitting receipt is not complete')
    need(receipt.get('selections')==280 and receipt.get('model_replays')==560
         and receipt.get('all_full_predictions_replayed') is True and receipt.get('exact_keyed_grid_verified') is True,'Full utility fitting/replay closure incomplete')
    planpath=root/'plan.json';plan=C.read(planpath);ph=B.sha(planpath)
    need(receipt.get('plan_sha256')==ph and plan.get('pilot') is False,'Full plan receipt binding differs')
    need(plan.get('sources_sha256')==manifest['producer_sources_sha256'],'Full producer source bindings differ')
    need(set(plan.get('panels',{}))==set(ENDPOINTS) and tuple(plan.get('selectors',[]))==SELECTORS
         and tuple(plan.get('evaluators',[]))==EVALUATORS and tuple(plan.get('budgets',[]))==BUDGETS
         and tuple(plan.get('seeds',[]))==SEEDS and plan.get('GPU') is False,'Full native scientific grid differs')
    selections={key:root/'selections'/key[0]/(key[1]+'__b%d__s%d.json'%key[2:]) for key in selection_grid()}
    models={key:root/'models'/key[0]/(key[1]+'__b%d__s%d__'%key[2:4]+key[4])/'result.json' for key in model_grid()}
    sc=normalized_catalog(receipt.get('selection_receipts_sha256'),selections.values())
    mc=normalized_catalog(receipt.get('model_receipts_sha256'),models.values())
    for (dataset,selector,budget,seed),p in selections.items():
        r=C.read(p)
        need(r.get('phase')=='complete' and r.get('plan_sha256')==ph
             and r.get('dataset')==dataset and r.get('selector')==selector and r.get('budget')==budget and r.get('seed')==seed,
             'Native selection identity differs')
        need(r.get('labels_supplied') is False and r.get('class_count_supplied') is False,
             'Representative selection read endpoint labels or known class count')
    for (dataset,selector,budget,seed,evaluator),p in models.items():
        r=C.read(p)
        need(r.get('phase')=='complete' and r.get('plan_sha256')==ph
             and tuple(r.get(k) for k in ('dataset','selector','budget','seed','evaluator'))==(dataset,selector,budget,seed,evaluator),
             'Native utility model identity differs')
        need(r.get('heldout_outcomes_read') is False and r.get('full_prediction_rows_replayed')==plan['panels'][dataset]['n_probe'],
             'Native utility outcome boundary/prediction population differs')
        need(r.get('selection_receipt_sha256')==sc[str(selections[(dataset,selector,budget,seed)].resolve())],
             'Model is bound to a different representative selection')
    hashes={**sc,**mc,str(path.resolve()):B.sha(path),str(planpath.resolve()):ph}
    return dict(phase='complete',stage='fit',selections=280,models=560,
                native_receipt_sha256=B.sha(path),outputs_sha256=hashes,
                execution_sha256=B.sha(ROOT/'execution.json'))


def native_score_ready(manifest):
    full=native_fit_ready(manifest)
    if not full:return False
    path=NATIVE/'full/COMPLETE.json'
    if not path.exists():return False
    receipt=C.read(path)
    if receipt.get('phase') in ('pending','running','in_progress'):return False
    need(receipt.get('phase')=='complete' and receipt.get('schema')=='synrfp37-clustering-utility-v2-complete'
         and receipt.get('selections')==280 and receipt.get('models')==560,'Native scored utility scope differs')
    need(receipt.get('plan_sha256')==B.sha(NATIVE/'full/plan.json'),'Native scored plan changed')
    catalog=receipt.get('outputs_sha256');checked_pins(catalog)
    cells=NATIVE_EXPORT/'heldout_cells.csv'
    need(str(cells.resolve()) in {str(Path(p).resolve()) for p in catalog},'Scored cells lack output binding')
    with cells.open(newline='') as f:records=list(csv.DictReader(f))
    keys=set()
    for row in records:
        key=(row['dataset'],row['selector'],int(row['budget']),int(row['seed']),row['evaluator'])
        need(key not in keys,'Duplicate heldout utility evaluation');keys.add(key)
        need(all(math.isfinite(float(row[m])) for m in ('mcc','macro_f1','accuracy')),'Nonfinite utility metric')
    need(keys==model_grid(),'Heldout utility keyed grid differs')
    return dict(phase='complete',stage='score',selections=280,models=560,
        native_receipt_sha256=B.sha(path),outputs_sha256={**catalog,str(path.resolve()):B.sha(path)},
        execution_sha256=B.sha(ROOT/'execution.json'))


def refresh_verified(stage,manifest):
    path=ROOT/'verified'/(stage+'.json')
    if path.exists():
        need(C.gate(dict(kind='receipt',path=str(path),equals=dict(phase='complete',stage=stage)),D),'Previously admitted completion disappeared')
        need(C.read(path)['execution_sha256']==B.sha(ROOT/'execution.json'),'Stage marker belongs to another execution')
        return True
    ready=native_fit_ready(manifest) if stage=='fit' else native_score_ready(manifest)
    if not ready:return False
    B.write(path,ready);return True


def scientific_failure(log):
    return bool(re.search(r'(?:ValueError|IntegrityError|AssertionError|ImportError|ModuleNotFoundError):|RuntimeError:[^\n]*(?:replay|validation|mismatch|differs|changed|digest|hash|identity|sealed|finite|shape)',log,re.I))


def ensure(job):
    props=C.props(job['unit']);C.attest_service(job,props)
    if props.get('ActiveState')=='failed':
        p=C.ROOT/'attempts'/(job['id']+'.json');prior=C.read(p) if p.exists() else {}
        log=C.failure_logs(job,props,prior)
        if props.get('Result')=='oom-kill' or scientific_failure(log):
            C.latch(job,'cgroup_oom' if props.get('Result')=='oom-kill' else 'scientific_validation_failure_no_retry',dict(unit=job['unit'],logs=log))
    try:return C.ensure(job,D)
    except subprocess.CalledProcessError as error:
        if error.returncode!=5 or list(error.cmd)!=['systemctl','--user','start',job['unit']]:raise
        p=C.props(job['unit'])
        if p.get('LoadState')!='not-found' or p.get('ActiveState') in C.ACTIVE:raise
        need(not (C.ROOT/'latches'/(job['id']+'.json')).exists(),'Refuse transient recovery after fatal worker latch')
        checked_pins(job['sources_sha256'])
        admitted=B.launch(job)
        return C.ensure(job,D) if admitted['phase'] in ('launched','already_active') else admitted


def frozen_owned(specs):
    frozen=[]
    for job in specs.values():
        props=C.props(job['unit'])
        if props.get('ActiveState') in C.ACTIVE:
            C.attest_service(job,props);C.freeze_or_thaw('freeze',job['unit']);frozen.append(job['unit'])
    return frozen


def cycle():
    manifest=seal()
    if manifest is None:return dict(phase='waiting_tested37v2_pilot_and_root_review',gpu_jobs=0)
    specs=jobs(manifest)
    for job in specs.values():C.attest_service(job,C.props(job['unit']))
    pressure=C.pressure(resource_snapshot(),list(specs.values()))
    if pressure['units'] or pressure['low_memory'] or pressure['low_disk']:return dict(phase='waiting_resource_recovery',pressure=pressure)
    decisions={}
    for stage in ('fit','score'):
        done=refresh_verified(stage,manifest)
        decisions[stage]=ensure(specs[stage])
        if (C.ROOT/'latches'/(specs[stage]['id']+'.json')).exists():
            return dict(phase='blocked_by_scientific_or_oom_failure',decisions=decisions,frozen_units=frozen_owned(specs),automatic_bypass=False)
        state=C.props(specs[stage]['unit'])
        if done and state.get('ActiveState') not in C.ACTIVE:
            need(state.get('ActiveState')!='failed','Completed native receipt accompanied a failed service')
            continue
        return dict(phase='running_utility_'+stage,decisions=decisions,gpu_jobs=0)
    result=dict(phase='complete',completed_utc=B.now(),selections=280,models=560,
        native_score_receipt_sha256=B.sha(NATIVE/'full/COMPLETE.json'),execution_sha256=B.sha(ROOT/'execution.json'),gpu_jobs=0,
        scope='Only reviewed37v2chemical utility; study38andnewscientifictrainingnotadmitted')
    B.write(ROOT/'COMPLETE.json',result);B.write(EXPORT/'COMPLETE.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--once',action='store_true');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.prepare:prepare();print('Controller source seal prepared; no worker launched');return
    with (ROOT/'controller.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'CONTROLLER_FAILED.json').exists():return
        try:
            while True:
                value=dict(cycle(),updated_utc=B.now());B.write(ROOT/'status.json',value);B.write(EXPORT/'status.json',value)
                if args.once or value['phase'] in ('complete','blocked_by_scientific_or_oom_failure'):return
                time.sleep(30)
        except Exception as error:
            manifest=ROOT/'execution.json';frozen=[]
            if manifest.exists():frozen=frozen_owned(jobs(C.read(manifest)))
            B.write(ROOT/'CONTROLLER_FAILED.json',dict(phase='failed',reason=repr(error),frozen_own_units=frozen,
                    scientific_artifacts_preserved=True,automatic_retry=False,updated_utc=B.now()))
            raise


if __name__=='__main__':main()
