"""Independently publish completed Morgan43 and Path44 metadata/code only.

Scientific gates stream declared artifact hashes and validate full keyed scalar
populations. Git uses an isolated index and compare-and-swap transactions; the
checked-out package source remains unchanged. No commit/push occurs at import.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path.home()/'.local/share/synrfp-runs/corrected_results_publication_v1'
ROOT43=ROOT.parent/'morgan_ecfp_correction_v1'
ROOT44=ROOT.parent/'task_preset_comparison_v1'
CORE=REPO.parent/'SynRFP-morgan-ecfp-v004'
BRANCH='exp/reproducible-benchmark'
CORE_BRANCH='fix/morgan-ecfp-0.0.4'
REMOTE='origin'
REMOTE_URL='git@github.com:TieuLongPhan/SynRFP.git'
PATH_CONFIG='synrfp-path-r2-b2048-parity-delta'
FORBIDDEN={'raw','weights','models','checkpoints','cache','caches','features','runs','__pycache__'}
SUFFIXES={'.py','.md','.json','.csv','.service','.toml'}
REPORT43_NAMES=('cells.csv','configuration_summary.csv','observed_ranks.csv','cross_panel_consistency.csv','paired_full_interactions.csv','panel_full_interactions.csv','metric_refusals.json','protocol.json','inputs.json','report.md','COMPLETE.json')
REPORT44_NAMES=('cells.csv','with_shared_control.csv','summary.csv','summary_with_shared_control.csv','protocol.json','inputs.json','COMPLETE.json')
EX43=REPO/'Experiments/43_morgan_ecfp_correction/publication_metadata/morgan_ecfp_correction_v1'
EX44=REPO/'Experiments/44_task_preset_comparison/publication_metadata/task_preset_comparison_v1'
OWN=('publisher.py','controller.py','test_publisher.py','test_controller.py','PUBLICATION_PROTOCOL.md','synrfp-corrected-results-publish46-v1.service')

# Explicit Python-only dependencies of the admitted43/44 science and the46
# watcher. Their results, data and optional legacy services are not published.
HELPERS=(
 'Experiments/25_followup_scheduler/budget.py','Experiments/26_ncs_continuous/controller.py',
 'Experiments/39_ncs_execution/controller.py','Experiments/42_knn_factorial/controller.py','Experiments/42_knn_factorial/resource85.py',
 'Experiments/20_focused_factorial/encode_factorial.py','Experiments/20_focused_factorial/analysis.py','Experiments/20_focused_factorial/per_task_analysis.py',
 'Experiments/22_all_endpoint_factorial/study.py','Experiments/22_all_endpoint_factorial/models.py','Experiments/22_all_endpoint_factorial/clustering.py',
 'Experiments/22_all_endpoint_factorial/gpu_worker.py','Experiments/22_all_endpoint_factorial/gpu_worker_runtime_v2.py',
 'Experiments/16_modes/classification_worker.py','Experiments/03_classification/mlp_gpu_queue.py',
 'Experiments/12_validation/07_clustering.py','Experiments/12_validation/vlib.py','Experiments/12_validation/10_ncs_extensions.py',
 'Experiments/12_validation/run_crxnfp_full.py','Experiments/12_validation/run_classification_r2_wide.py','Experiments/12_validation/crxnfp_full_worker.py',
 'Experiments/13_yield/yield_retrain_cpu.py',
)
PATH_OPERATIONAL43=('priority_policy_v2.py','controller_priority_v2.py','test_controller_priority_v2.py','synrfp-morgan43-priority-v2.service',
 'controller.py','test_controller.py','EXECUTION.md','synrfp-morgan43-v1.service')



def module(name,path):
    q=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(q);sys.modules[name]=m;q.loader.exec_module(m);return m


P=module('publication46_frozen_private_index',REPO/'Experiments/34_final_comparison/publish_main_results.py')
need,read,write,sha,now=P.need,P.read,P.write,P.sha,P.now
git=P.git


class HashCache:
    def __init__(self):self.values={}
    def verify(self,path,expected):
        p=Path(path).resolve();v=p.stat();sig=(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
        if str(p) not in self.values or self.values[str(p)][0]!=sig:
            actual=sha(p);v=p.stat();need(sig==(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns),'File changed during hash');self.values[str(p)]=(sig,actual)
        need(self.values[str(p)][1]==expected,'Declared artifact digest changed: '+str(p));return expected


def safe_path(repo,relative):
    relative=Path(relative);need(not relative.is_absolute() and '..'not in relative.parts,'Publication path escapes checkout')
    need(not set(relative.parts)&FORBIDDEN and relative.suffix in SUFFIXES,'Heavyweight/nonmetadata publication path')
    p=Path(repo)/relative;need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(Path(repo).resolve()),'Missing/symlinked publication path')
    need(p.stat().st_size<=20*1024**2,'Metadata file exceeds20MiB');return p


# Reuse the reviewed transaction body; change only its literal commit message
# and broaden the safe metadata extension set to permit package pyproject.toml.
P.safe_path=safe_path
_tree=ast.parse(inspect.getsource(P.commit_snapshot));_changed=0
for _n in ast.walk(_tree):
    if isinstance(_n,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='message'for t in _n.targets):
        _n.value=ast.Name(id='PUBLICATION_MESSAGE',ctx=ast.Load());_changed+=1
need(_changed==1,'Private-index message boundary changed')
ast.fix_missing_locations(_tree);exec(compile(_tree,str(REPO/'Experiments/34_final_comparison/publish_main_results.py')+'[46-message-only]','exec'),P.__dict__)


def source_pins():
    paths=[HERE/n for n in OWN]+[REPO/'Experiments/34_final_comparison/publish_main_results.py',REPO/'Experiments/34_final_comparison/publish_main_runtime_v2.py']
    return {str(p.resolve()):sha(p)for p in paths}


def core_paths(ready):
    expected={'synrfp/tokenizers/morgan.py','pyproject.toml','CHANGELOG.md'}
    for p in ready['sources_sha256']:
        q=Path(p).resolve();need(q.is_relative_to(CORE.resolve()),'Core pin outside corrected checkout')
        rel=q.relative_to(CORE.resolve()).as_posix()
        if rel.startswith('tests/tokenizers/test_morgan')and rel.endswith('.py'):expected.add(rel)
    need('tests/tokenizers/test_morgan_ecfp.py'in expected,'Corrected ECFP controls missing');return sorted(expected)


def csv_rows(path):
    with Path(path).open(newline='')as f:
        reader=csv.DictReader(f)
        need(not set(reader.fieldnames or [])&{'rxn','reaction','smiles','predictions','observed','assignments','centers','X'},'Raw data/prediction arrays in scalar export')
        return list(reader)


def verify_catalog(catalog,cache):
    need(isinstance(catalog,dict) and bool(catalog),'Declared completion hash catalog missing')
    for p,h in catalog.items():need(Path(p).is_absolute(),'Local completion catalog must use absolute paths');cache.verify(p,h)


def check_morgan_header(r):
    need(r.get('schema')=='morgan43-full-correction-complete-v1'and r.get('phase')=='complete'
      and r.get('tokenizer_profile')=='synrfp-ecfp-count-v1'and r.get('all_keyed_closures_verified')is True
      and r.get('corrected_mlp_cells')==1680 and r.get('corrected_clustering_cells')==1440
      and r.get('combined_classification_cells')==5040 and r.get('combined_clustering_cells')==4320,'Morgan43 full closure incomplete')


def check_path_header(r):
    need(r.get('schema')=='synrfp-task-preset-comparison-complete-v1'and r.get('phase')=='complete'and r.get('task')=='clustering'
      and r.get('candidate')==PATH_CONFIG and r.get('candidate_cells')==240 and r.get('baseline_cells')==1200
      and r.get('primary_cells')==1440 and r.get('cells_with_shared_control')==1680 and r.get('initialization_seeds')==10
      and r.get('all_keyed_cells_verified')is True and r.get('all_retained_artifact_hashes_verified')is True
      and r.get('classification_deferred')is True,'Path44 ten-seed closure incomplete')



def keyed_grid(rows,expected,keys):
    actual=[tuple(int(r[k])if k in ('seed','fold')else r[k]for k in keys)for r in rows]
    need(len(actual)==len(set(actual))and set(actual)==set(expected),'Missing/duplicate/foreign keyed scalar crossing')
    return actual


def morgan_gate(cache=None):
    cache=cache or HashCache();top=read(ROOT43/'COMPLETE.json');check_morgan_header(top);verify_catalog(top['outputs_sha256'],cache)
    plan=read(ROOT43/'plan.json');need(top['plan_sha256']==sha(ROOT43/'plan.json')==read(ROOT43/'plan_seal.json')['sha256'],'Morgan plan/seal mismatch')
    ready=read(ROOT43/'CORE_READY.json');need(ready.get('phase')=='complete'and ready.get('tokenizer_profile')=='synrfp-ecfp-count-v1'and ready.get('package_version')=='0.0.4'and ready.get('tests_failed')==0,'Corrected core not complete')
    verify_catalog(ready['sources_sha256'],cache);verify_catalog(ready['outputs_sha256'],cache)
    old=read(ROOT43.parent/'all_endpoint_factorial_full_v1/plan.json');configs=old['configs'];proof={str(ROOT43/'COMPLETE.json'):sha(ROOT43/'COMPLETE.json'),str(ROOT43/'CORE_READY.json'):sha(ROOT43/'CORE_READY.json')}
    for task,scope,total in [('classification',plan['classification_datasets'],5040),('clustering',plan['clustering_datasets'],4320)]:
        local=ROOT43/'report'/task/'COMPLETE.json';r=read(local)
        need(r.get('phase')=='complete'and r.get('cells')==total and r.get('all_keyed_cells_verified')is True
          and r.get('all_retained_artifact_hashes_verified')is True and r.get('source_and_feature_closure_verified')is True
          and r.get('interaction_orders')==[1,2,3,4]and r.get('plan_sha256')==top['plan_sha256'],'Morgan task report closure differs')
        verify_catalog(r['outputs_sha256'],cache);rows=csv_rows(EX43/task/'cells.csv');key='fold'if task=='classification'else'seed'
        expected={(d,c,i)for d in scope for c in configs for i in range(5)};actual=[(x['dataset'],x['config'],int(x[key]))for x in rows]
        need(len(actual)==total,'Morgan report count differs');keyed_grid(rows,expected,('dataset','config',key))
        for row in rows:
            c=row['config'];fresh=configs[c]['tokenizer']=='morgan';need(row['tokenizer_profile']==('synrfp-ecfp-count-v1'if fresh else'study22-original-wl-path'),'Historical Morgan relabelled as corrected')
            need(row['result_origin']==('fresh-study43-ECFP'if fresh else 'reused-study22-WL-Path'),'Wrong result origin')
            base=ROOT43 if fresh else ROOT43.parent/'all_endpoint_factorial_full_v1'
            if task=='classification':p=base/'models'/('classification__'+row['dataset']+'__f'+str(int(row[key]))+'__'+c)/'result.json'
            else:p=base/('artifacts/clustering'if fresh else'clustering')/row['dataset']/c/(str(int(row[key]))+'.json')
            need(str(p.resolve())in r['outputs_sha256'],'Native result absent from declared closure');cache.verify(p,r['outputs_sha256'][str(p.resolve())]);native=read(p)
            need(all(abs(float(row[m])-float(native[m]))<=1e-12 for m in (('mcc','accuracy','macro_f1')if task=='classification'else('ami','ari','nmi'))),'Scalar/native result mismatch')
        proof[str(local)]=sha(local)
    return dict(component='morgan',phase='complete',logical_corrected_cells=3120,combined_report_cells=9360,tokenizer_profile='synrfp-ecfp-count-v1',scientific_receipts_sha256=proof,core_paths=core_paths(ready),core_ready_sha256=sha(ROOT43/'CORE_READY.json'))


def path_gate(cache=None):
    cache=cache or HashCache();top=read(ROOT44/'COMPLETE.json');check_path_header(top);verify_catalog(top['outputs_sha256'],cache)
    native_plan=ROOT44/'clustering/plan.json';need(top['scientific_plan_sha256']==sha(native_plan),'Path native scientific plan changed')
    out=EX44/'clustering';exported=read(out/'COMPLETE.json');check_path_header(exported)
    need(exported['scientific_plan_sha256']==top['scientific_plan_sha256']and exported['decision_sha256']==top['decision_sha256'],'Path exported/native completion differs')
    for name,h in exported['outputs_sha256'].items():need(name in REPORT44_NAMES,'Unexpected Path output');cache.verify(out/name,h)
    rows=csv_rows(out/'cells.csv');both=csv_rows(out/'with_shared_control.csv')
    endpoints=('schneider_ub','schneider_b','uspto50k_ub','uspto50k_b','syntemp_r0','mech_1','mech_2','tpl_ub','tpl_b','ecreact_1','ecreact_2','ecreact_3')
    methods=('drfp','rdkit_struct','rdkit_diff','rxnfp_mlm','crxnfp','synrfp_task_candidate')
    for values,allowed,count in ((rows,methods,1440),(both,methods+('synrfp_shared',),1680)):
        actual=[(r['dataset'],r['population'],int(r['seed']),r['method'])for r in values]
        expected={(d,p,s,m)for d in endpoints for p in ('all_rows','shared_valid_all_six')for s in range(10)for m in allowed}
        need(len(actual)==count,'Path report count differs');keyed_grid(values,expected,('dataset','population','seed','method'))
        for row in values:
            if row['method']=='synrfp_task_candidate':need(row['configuration']==PATH_CONFIG,'Mixed Path configurations')
            p=row.get('source_result_path')or row.get('result_path');h=row.get('source_result_sha256')or row.get('result_sha256')
            need(p and h,'Scalar row lacks native digest authority');cache.verify(p,h)
    return dict(component='path',phase='complete',candidate_cells=240,baseline_cells=1200,primary_cells=1440,cells_with_shared_control=1680,initialization_seeds=10,
      scientific_receipts_sha256={str(ROOT44/'COMPLETE.json'):sha(ROOT44/'COMPLETE.json'),str(out/'COMPLETE.json'):sha(out/'COMPLETE.json')})


def source_files(folder):
    return sorted(p.relative_to(REPO).as_posix()for p in (REPO/folder).iterdir()if p.is_file()and p.suffix in {'.py','.md','.service'}and p.name not in ('classification.py','test_classification.py'))


def main_paths(component,config):
    paths=list(config['component_sources'][component])+list(config['publisher_paths'])
    if component=='morgan':
        paths +=[(EX43/task/n).relative_to(REPO).as_posix()for task in ('classification','clustering')for n in REPORT43_NAMES]
        paths +=[(EX43/n).relative_to(REPO).as_posix()for n in ('CORE_READY.json','plan.json','plan_seal.json','CODE_READY.json','ENVIRONMENT_READY.json','ROOT_REVIEW.json','RELEASE_PROVENANCE.json')]
    else:
        paths +=[(EX44/'clustering'/n).relative_to(REPO).as_posix()for n in REPORT44_NAMES]
        paths +=[(EX44/n).relative_to(REPO).as_posix()for n in ('decision.json','plan.json','PLAN_READY.json','ROOT_REVIEW.json','PUBLICATION_SCOPE.json')]
    need(not any(p.startswith('synrfp/')or p=='pyproject.toml'for p in paths),'Main package source must remain untouched')
    return sorted(set(paths))


def main_core_pins():
    paths=git(REPO,'ls-files','synrfp','pyproject.toml').splitlines()
    return {p:sha(REPO/p)for p in paths if (REPO/p).is_file()}


def prepare():
    need(P.branch(REPO)==BRANCH and P.branch(CORE)==CORE_BRANCH,'Unexpected publication checkout branches')
    for repo in (REPO,CORE):need(git(repo,'remote','get-url','--push',REMOTE).strip()==REMOTE_URL,'Unexpected push remote')
    publisher_paths=['Experiments/46_publish_corrected_results/'+n for n in OWN]
    library_helpers=sorted(p.relative_to(REPO).as_posix()for p in (REPO/'Experiments/synrfp_exp').glob('*.py'))
    helpers=list(HELPERS)+library_helpers
    sources={'morgan':sorted(set(source_files('Experiments/43_morgan_ecfp_correction')+helpers)),
      'path':sorted(set(source_files('Experiments/44_task_preset_comparison')+helpers+
        ['Experiments/43_morgan_ecfp_correction/'+n for n in PATH_OPERATIONAL43]))}
    for paths in sources.values():
        for relative in paths:safe_path(REPO,relative)
    value=dict(schema='synrfp-independent-corrected-publication-v1',phase='prepared',branch=BRANCH,core_branch=CORE_BRANCH,remote=REMOTE,remote_url=REMOTE_URL,
      initial_head=git(REPO,'rev-parse','HEAD').strip(),initial_core_head=git(CORE,'rev-parse','HEAD').strip(),publisher_paths=publisher_paths,
      publisher_sources_sha256=source_pins(),component_sources=sources,component_sources_sha256={p:sha(REPO/p)for paths in sources.values()for p in paths},
      original_main_core_sha256=main_core_pins(),authorization='User explicitly requested independent commit/push when Morgan43 and Path44 finish on20October2026',
      weights_local_only=True,force_push=False,private_index=True,independent_components=True,deferred_classification_excluded=True)
    path=ROOT/'config.json'
    if path.exists():
        old=read(path);need(all(old[k]==value[k]for k in value if k not in ('initial_head','initial_core_head')),'Publisher policy/source changed');return old
    write(path,value);return value


def verify_setup(config):
    need(P.branch(REPO)==BRANCH and P.branch(CORE)==CORE_BRANCH,'Publication branch changed')
    for repo in (REPO,CORE):need(git(repo,'remote','get-url','--push',REMOTE).strip()==config['remote_url'],'Push remote changed')
    for p,h in config['publisher_sources_sha256'].items():need(sha(p)==h,'Publisher changed after review')
    for p,h in config['component_sources_sha256'].items():need(sha(REPO/p)==h,'Scientific source changed after publication review')
    for p,h in config['original_main_core_sha256'].items():need(sha(REPO/p)==h,'Original package source changed; sealed jobs must remain valid')


def remote_head(repo,branch):
    value=git(repo,'ls-remote',REMOTE,'refs/heads/'+branch).split();return value[0]if value else None


def fetch_tip(repo,branch):
    tip=remote_head(repo,branch)
    if tip:git(repo,'fetch','--no-tags',REMOTE,branch);need(remote_head(repo,branch)==tip,'Remote changed while fetching; retry')
    return tip


def ancestor(repo,a,b):
    r=subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',a,b],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    need(r.returncode in (0,1),'Cannot determine Git ancestry');return r.returncode==0



def remote_paths_compatible(repo,approved_head,tip,paths,hashes):
    if tip is None:return
    common=git(repo,'merge-base',approved_head,tip).strip()
    for relative in paths:
        before=git(repo,'ls-tree',common,'--',relative).strip()
        after=git(repo,'ls-tree',tip,'--',relative).strip()
        if before==after:continue
        # Identical desired bytes may already have been published concurrently;
        # a different edit to an owned path is never overwritten automatically.
        if after:
            blob=git(repo,'show',tip+':'+relative).encode()
            if hashlib.sha256(blob).hexdigest()==hashes[relative]:continue
        raise ValueError('Remote edited an approved publication path: '+relative)


def private_commit(repo,base,paths,hashes,directory,message):
    directory.mkdir(parents=True,exist_ok=True);index=directory/'rebase.index';index.unlink(missing_ok=True)
    git(repo,'read-tree',base,index=index);git(repo,'add','--',*paths,index=index)
    changed=set(git(repo,'diff','--cached','--name-only',base,index=index).splitlines());need(changed<=set(paths),'Private rebase included unapproved paths')
    for p,h in hashes.items():
        need(sha(safe_path(repo,p))==h,'Snapshot changed while rebasing')
        blob=git(repo,'show',':'+p,index=index).encode();need(hashlib.sha256(blob).hexdigest()==h,'Private rebase blob differs')
    if not changed:return dict(base=base,commit=base,files_sha256=hashes,branch=P.branch(repo),local_head_preserved=True,already_present=True)
    tree=git(repo,'write-tree',index=index).strip();commit=git(repo,'commit-tree',tree,'-p',base,input=message.encode()).strip()
    return dict(base=base,commit=commit,tree=tree,files_sha256=hashes,branch=P.branch(repo),local_head_preserved=True,already_present=False)


def commit_and_push(repo,branch,paths,hashes,directory,message,initial_head):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);pending=directory/'COMMITTED.json'
    if pending.exists():tx=read(pending)
    else:
        need(P.branch(repo)==branch,'Publication branch changed')
        tip=fetch_tip(repo,branch);head=git(repo,'rev-parse','HEAD').strip()
        remote_paths_compatible(repo,head,tip,paths,hashes)
        if tip==head or (tip is None and head==initial_head):
            P.PUBLICATION_MESSAGE=message;tx=P.commit_snapshot(repo,paths,hashes,directory,branch);tx['local_head_preserved']=False
        else:
            # Rebase only the approved snapshot through commit-tree. Dirty user
            # work/index and the sealed checkout remain byte-for-byte untouched.
            tx=private_commit(repo,tip or initial_head,paths,hashes,directory,message)
        tx['approved_head']=head
        write(pending,tx)
    approved_head=tx.get('approved_head',tx['base'])
    for attempt in range(4):
        need(P.branch(repo)==branch,'Publication branch changed before push')
        commit=tx['commit'];tip=fetch_tip(repo,branch)
        remote_paths_compatible(repo,approved_head,tip,paths,hashes)
        if tip==commit:break
        if tip is not None and not ancestor(repo,tip,commit):
            tx=private_commit(repo,tip,paths,hashes,directory/('remote-rebase-'+str(attempt)),message);tx['approved_head']=approved_head;write(pending,tx);commit=tx['commit']
            if tip==commit:break
        try:git(repo,'push',REMOTE,commit+':refs/heads/'+branch)
        except RuntimeError:
            if attempt==3:raise
            continue
        if remote_head(repo,branch)==commit:break
    need(remote_head(repo,branch)==tx['commit'],'Remote publication not confirmed')
    for p,h in hashes.items():need(hashlib.sha256(git(repo,'show',tx['commit']+':'+p).encode()).hexdigest()==h,'Remote-confirmed commit blob differs')
    result=dict(tx,phase='complete',remote=REMOTE,remote_head=tx['commit'],published_utc=now(),force_push=False,weights_published=False)
    write(directory/'PUBLISHED.json',result);return result


def copy_metadata(source,target):
    source,target=Path(source),Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():need(target.read_bytes()==source.read_bytes(),'Existing publication metadata differs')
    else:
        tmp=target.with_name(target.name+'.tmp');tmp.write_bytes(source.read_bytes());tmp.replace(target)



def path_metadata_bindings():
    plan_path=ROOT44/'clustering/plan.json';plan=read(plan_path);decision=Path(plan['decision_path'])
    need(decision.resolve().is_relative_to((ROOT44/'decisions').resolve()),'Path decision outside declared namespace')
    need(sha(decision)==plan['decision_sha256'],'Frozen Path decision digest changed')
    return [(decision,'decision.json'),(plan_path,'plan.json'),(ROOT44/'clustering/PLAN_READY.json','PLAN_READY.json'),(ROOT44/'ROOT_REVIEW.json','ROOT_REVIEW.json')]


def publish(component):
    need(component in ('morgan','path'),'Unknown publication component');config=read(ROOT/'config.json');verify_setup(config)
    review=read(ROOT/'ROOT_REVIEW.json');ready=read(ROOT/'CODE_READY.json')
    need(review.get('schema')=='publication46-root-reviewed-v1'and review.get('phase')=='reviewed'
      and review.get('config_sha256')==sha(ROOT/'config.json')and set(review.get('components',[]))=={'morgan','path'}
      and ready.get('phase')=='complete'and ready.get('tests_passed',0)>0
      and review.get('code_ready_sha256')==sha(ROOT/'CODE_READY.json')
      and review.get('sources_sha256')==ready.get('sources_sha256'),'Publication needs exact tested root review')
    verify_catalog(review['sources_sha256'],HashCache())
    directory=ROOT/component
    if (directory/'PUBLISHED.json').exists():return read(directory/'PUBLISHED.json')
    cache=HashCache();gate=(morgan_gate if component=='morgan'else path_gate)(cache);write(directory/'GATE.json',gate)
    code_result=None
    if component=='morgan':
        ready=read(ROOT43/'CORE_READY.json');paths=core_paths(ready);hashes={p:ready['sources_sha256'][str((CORE/p).resolve())]for p in paths}
        code_result=commit_and_push(CORE,CORE_BRANCH,paths,hashes,directory/'core-code','Correct graph-native Morgan ECFP count construction (0.0.4)\n',config['initial_core_head'])
        for n in ('CORE_READY.json','plan.json','plan_seal.json','CODE_READY.json','ENVIRONMENT_READY.json','ROOT_REVIEW.json'):copy_metadata(ROOT43/n,EX43/n)
        release=dict(schema='synrfp-ecfp-v004-release-provenance-v1',package_version='0.0.4',tokenizer_profile='synrfp-ecfp-count-v1',code_branch=CORE_BRANCH,
          code_commit=code_result['commit'],code_url='https://github.com/TieuLongPhan/SynRFP/commit/'+code_result['commit'],core_ready_sha256=sha(ROOT43/'CORE_READY.json'),
          source_files_sha256=hashes,main_checkout_package_source_changed=False,scientific_gate=gate)
        write(EX43/'RELEASE_PROVENANCE.json',release)
    else:
        for src,name in path_metadata_bindings():copy_metadata(src,EX44/name)
        write(EX44/'PUBLICATION_SCOPE.json',dict(component='path',task='clustering',candidate=PATH_CONFIG,initialization_seeds=10,
          classification_deferred=True,main_shared_control_preserved=True,scientific_gate=gate,weights_local_only=True))
    paths=main_paths(component,config);hashes={p:sha(safe_path(REPO,p))for p in paths}
    write(directory/'SNAPSHOT.json',dict(component=component,files_sha256=hashes,scientific_gate=gate,prepared_utc=now()))
    verify_setup(config);need((morgan_gate if component=='morgan'else path_gate)(cache)==gate,'Scientific completion changed during snapshot')
    result=commit_and_push(REPO,BRANCH,paths,hashes,directory/'main','Publish '+('corrected Morgan ECFP factorial results'if component=='morgan'else'matched ten-seed Path clustering comparison')+'\n',config['initial_head'])
    proof=dict(schema='synrfp-independent-component-published-v1',phase='complete',component=component,commit=result['commit'],branch=BRANCH,remote=REMOTE,
      files_sha256=hashes,scientific_gate=gate,core_code_commit=code_result['commit']if code_result else None,weights_published=False,other_component_waited_for=False,published_utc=now())
    write(directory/'PUBLISHED.json',proof)
    log=REPO/'SS_LOG.md';line='\n'+component+' results published independently: '+result['commit']+'; '+now()+'. Weights remain local.\n'
    if log.exists()and result['commit']not in log.read_text():
        with log.open('a')as f:f.write(line)
    return proof


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--component',choices=['morgan','path']);args=a.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'publication.lock').open('a+')as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare:prepare()
        else:need(args.component,'Choose prepare or component');publish(args.component)
