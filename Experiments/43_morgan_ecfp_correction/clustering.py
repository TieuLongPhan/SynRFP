"""Exact original full-row binary MiniBatchKMeans, corrected Morgan inputs only."""
from pathlib import Path
import numpy as np
import pandas as pd
import study as S
import encode as E

C=S.module('morgan43_original_clustering',S.OLD_HERE/'clustering.py',S.legacy_shim())
C.output=S.result_path
_data=C.data
CACHE=S.HashCache()

def data(plan,task):
    source=plan['panels'][task['dataset']]['source']
    if E.feature_ready(Path(plan['root']),plan,source,cache=CACHE) is None:return None
    return _data(plan,task)

C.data=data

def verify_result(root,plan,task,record=None,deep=False,cache=None):
    path=S.result_path(root,task)
    if record is None:
        if not path.exists():return None
        record=S.read(path)
    S.need(record['phase']=='complete' and all(record[k]==task[k] for k in task),'Changed clustering task')
    S.need(record['labels_used_for_fit_or_selection'] is False and record['params']==plan['clustering_params'],'Changed label-blind clustering recipe')
    panel=plan['panels'][task['dataset']];src=plan['sources'][panel['source']]
    S.need(record['panel_sha256']==S.sha(panel['table']) and record['feature_sha256']==S.sha(src['features'][task['config']]),'Changed clustering feature/panel')
    C.verify_result(root,plan,task,record,replay=deep)
    f=pd.read_csv(panel['table'],dtype={'row_id':str})
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
        S.need(np.array_equal(z['row_ids'].astype(str),f.row_id.to_numpy(dtype=str)) and np.array_equal(z['source_positions'],f.source_position.to_numpy()),'Changed clustering row identity')
    return record

def fit(root,plan,task,threads=2):
    E.disk_guard(root)
    if S.result_path(root,task).exists():return verify_result(root,plan,task)
    result=C.fit(root,plan,task,threads)
    if result:verify_result(root,plan,task,deep=True)
    return result
