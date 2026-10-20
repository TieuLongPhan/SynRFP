"""Fresh publication policy including the bounded Path report-root repair.

The sealed v1/v2 publishers and all prior receipts remain unchanged.  The
scientific cells, metrics, private-index transaction and independent triggers
are inherited; only the reviewed source allowlist/root and repair provenance
are extended.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
q=importlib.util.spec_from_file_location('publication46_v3_previous_policy',HERE/'publisher_runtime_v2.py')
V=importlib.util.module_from_spec(q);sys.modules[q.name]=V;q.loader.exec_module(V)
P=V.P
LEGACY_ROOT=V.ROOT
MUTEX_ROOT=V.LEGACY_ROOT
ROOT=LEGACY_ROOT.with_name('corrected_results_publication_v3')
P.ROOT=ROOT;V.ROOT=ROOT
FILES=('publisher_runtime_v3.py','controller_runtime_v3.py','test_runtime_v3.py','synrfp-corrected-results-publish46-v3.service')
P.OWN=tuple(dict.fromkeys(P.OWN+FILES))
RECOVERY_FILES=('report_recovery_v4.py','test_report_recovery_v4.py')
SCHEMA='synrfp-independent-corrected-publication-v3'
V.SCHEMA=SCHEMA
ORIGINAL_MAIN_PATHS=V.main_paths
ORIGINAL_PATH_GATE=P.path_gate


def recovery_sources():
    return {'Experiments/44_task_preset_comparison/'+name:
            P.sha(P.safe_path(P.REPO,'Experiments/44_task_preset_comparison/'+name))
            for name in RECOVERY_FILES}


def main_paths(component,config):
    # The previous wrapper's version discriminator concerns its partition
    # exports.  Reuse that unchanged operation under the new reviewed policy.
    adapted={**config,'publication_runtime_version':2}
    return ORIGINAL_MAIN_PATHS(component,adapted)


P.main_paths=main_paths


def path_gate(cache=None):
    cache=cache or P.HashCache()
    result=ORIGINAL_PATH_GATE(cache)
    config=P.read(ROOT/'config.json')
    authority=P.read(P.ROOT44/'COMPLETE.json')['outputs_sha256']
    for relative,digest in config['required_path_recovery_sources_sha256'].items():
        path=(P.REPO/relative).resolve()
        P.need(authority.get(str(path))==digest,'Path native closure omits the reviewed report recovery source')
        cache.verify(path,digest)
    return result


P.path_gate=path_gate
morgan_gate=P.morgan_gate


def prepare():
    recovery=recovery_sources();path=ROOT/'config.json'
    if path.exists():
        config=P.read(path)
        P.need(config.get('schema')==SCHEMA and config.get('publication_runtime_version')==3
          and config.get('required_path_recovery_sources_sha256')==recovery,'Version3 publication policy/source differs')
        P.verify_setup(config);return config
    config=V.prepare()
    P.need(set(recovery)<=set(config['component_sources']['path']),'Path report recovery omitted from publication allowlist')
    config.update(schema=SCHEMA,publication_runtime_version=3,supersedes_publication_root=str(LEGACY_ROOT),
      legacy_config_sha256=P.sha(LEGACY_ROOT/'config.json'),required_path_recovery_sources_sha256=recovery,
      scientific_cells_and_scoring_unchanged=True,repair_scope='Resolve baseline artifact keys against their declared original roots')
    # If an earlier component finished during the governor handoff, preserve its
    # immutable proof rather than asking the new watcher to publish it again.
    previous_legacy=V.LEGACY_ROOT
    try:
        V.LEGACY_ROOT=LEGACY_ROOT
        config['old_completed_publication_proofs_sha256'].update(V.adopt_completed_legacy())
    finally:V.LEGACY_ROOT=previous_legacy
    P.write(path,config);return config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--component',choices=['morgan','path']);args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);MUTEX_ROOT.mkdir(parents=True,exist_ok=True)
    with (MUTEX_ROOT/'publication.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare:prepare()
        else:
            P.need(args.component,'Choose prepare or component')
            config=P.read(ROOT/'config.json')
            P.need(config.get('publication_runtime_version')==3 and config['required_path_recovery_sources_sha256']==recovery_sources(),'Wrong or changed report-recovery publication runtime')
            P.publish(args.component)


if __name__=='__main__':main()
