"""Isolated publication policy for the reviewed five-lane Path44 runtime.

The sealed46v1 implementation, configuration and receipts remain untouched.
Only policy capture/dispatch paths change; scientific gates and Git transactions
are reused verbatim from that version.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
q=importlib.util.spec_from_file_location('publication46_v2_frozen_publisher',HERE/'publisher.py')
P=importlib.util.module_from_spec(q);sys.modules[q.name]=P;q.loader.exec_module(P)
LEGACY_ROOT=P.ROOT
ROOT=LEGACY_ROOT.with_name('corrected_results_publication_v2')
P.ROOT=ROOT
FILES=('publisher_runtime_v2.py','controller_runtime_v2.py','test_runtime_v2.py','synrfp-corrected-results-publish46-v2.service')
P.OWN=tuple(dict.fromkeys(P.OWN+FILES))
V3_FILES=('parallel_v3.py','worker_parallel_v3.py','controller_parallel_v3.py','test_parallel_v3.py','PARALLEL_V3.md',
 'synrfp-task-preset44-parallel-v3.service','report_parallel_v3.py','test_report_parallel_v3.py')
SCHEMA='synrfp-independent-corrected-publication-v2'
PROVENANCE_FILES=('plan.json','plan_seal.json')
ORIGINAL_MAIN_PATHS=P.main_paths


def require_v3_sources():
    result={}
    for name in V3_FILES:
        relative='Experiments/44_task_preset_comparison/'+name
        result[relative]=P.sha(P.safe_path(P.REPO,relative))
    return result


def adopt_completed_legacy():
    """Copy only verified publication proofs; never modify an old receipt."""
    adopted={}
    for component in ('path','morgan'):
        source=LEGACY_ROOT/component/'PUBLISHED.json'
        if not source.exists():continue
        r=P.read(source)
        P.need(r.get('phase')=='complete'and r.get('component')==component and r.get('branch')==P.BRANCH
          and r.get('weights_published')is False and len(r.get('commit',''))==40,'Invalid prior independent publication proof')
        P.copy_metadata(source,ROOT/component/'PUBLISHED.json');adopted[str(source.resolve())]=P.sha(source)
    return adopted



def main_paths(component,config):
    paths=ORIGINAL_MAIN_PATHS(component,config)
    if component=='path'and config.get('publication_runtime_version')==2:
        authority=P.read(P.ROOT44/'COMPLETE.json')['outputs_sha256']
        for name in PROVENANCE_FILES:
            source=P.ROOT44/'parallel_v3'/name;key=str(source.resolve())
            P.need(key in authority,'Five-lane partition provenance missing from native closure')
            P.need(P.sha(source)==authority[key],'Five-lane partition provenance changed')
            target=P.EX44/'parallel_v3'/name;P.copy_metadata(source,target)
            paths.append(target.relative_to(P.REPO).as_posix())
    return sorted(set(paths))


P.main_paths=main_paths


def prepare():
    v3=require_v3_sources();path=ROOT/'config.json'
    if path.exists():
        config=P.read(path);P.need(config.get('schema')==SCHEMA and config.get('required_path_v3_sources_sha256')==v3,'Version2 publication policy/source differs')
        P.verify_setup(config);return config
    config=P.prepare()
    P.need(set(v3)<=set(config['component_sources']['path']),'Five-lane Path sources omitted from publication allowlist')
    config.update(schema=SCHEMA,publication_runtime_version=2,supersedes_publication_root=str(LEGACY_ROOT),
      legacy_config_sha256=P.sha(LEGACY_ROOT/'config.json'),required_path_v3_sources_sha256=v3,
      old_completed_publication_proofs_sha256=adopt_completed_legacy(),scientific_gates_unchanged=True,
      path_v3_provenance_exports=['Experiments/44_task_preset_comparison/publication_metadata/task_preset_comparison_v1/parallel_v3/'+n for n in PROVENANCE_FILES])
    P.write(path,config);return config


morgan_gate=P.morgan_gate
path_gate=P.path_gate
publish=P.publish
source_pins=P.source_pins


def main():
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--component',choices=['morgan','path']);args=a.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    # Both versions share the same publication mutex, even during an unexpected
    # service restart. Its physical inode is preserved; no lease is recreated.
    LEGACY_ROOT.mkdir(parents=True,exist_ok=True)
    with (LEGACY_ROOT/'publication.lock').open('a+')as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare:prepare()
        else:
            P.need(args.component,'Choose prepare or component')
            config=P.read(ROOT/'config.json');P.need(config.get('publication_runtime_version')==2,'Wrong publication runtime')
            P.need(config['required_path_v3_sources_sha256']==require_v3_sources(),'Five-lane Path source changed')
            publish(args.component)

if __name__=='__main__':main()
