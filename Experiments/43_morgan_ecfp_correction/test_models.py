from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import study as S
import models as M

def fixture(directory):
    root=Path(directory);n=40;c='synrfp-morgan-r2-b2048-parity-delta'
    table=root/'panel.csv';pd.DataFrame(dict(row_id=['id'+str(i) for i in range(n)],source_position=np.arange(n),y=np.arange(n)%2)).to_csv(table,index=False)
    split=root/'split.npy';code=np.zeros(n,np.int8);code[::5]=2;code[1::5]=1;np.save(split,code)
    fp=root/'feature.npz';S.npz(fp,X=np.packbits(np.random.RandomState(3).randint(0,2,(n,2048)).astype(np.uint8),axis=1),ok=np.ones(n,bool),n_bits=np.int64(2048),source_positions=np.arange(n),row_ids=np.asarray(['id'+str(i) for i in range(n)]))
    plan=dict(root=str(root),panels={'toy':dict(table=str(table),split_paths={'0':str(split)},family='classification',source='toy',n=n)},
        sources={'toy':dict(features={c:str(fp)},feature_receipt=str(root/'features_COMPLETE.json'))},params={'classification':dict(hidden=[512],alpha=.0001,max_iter=200,batch_size=256)},core_ready_sha256='c'*64,
        clustering_params=dict(batch_size=8192,n_init=3,max_iter=100,reassignment_ratio=.01,tol=0.,max_no_improvement=10))
    S.write(root/'plan.json',plan);S.write(root/'features_COMPLETE.json',dict(files={fp.name:S.sha(fp)}))
    return root,plan,dict(task='classification',dataset='toy',config=c,fold=0),code

class ModelTests(unittest.TestCase):
    def test_original_initialization_preserved(self):
        direct=S.module('morgan43_initialization_oracle',S.OLD_HERE/'models.py',S.legacy_shim())
        y=np.arange(100)%3;p=dict(hidden=[512]);a,x=M.M.initialization(y,p,2);b,z=direct.initialization(y,p,2)
        self.assertTrue(np.array_equal(a,b));self.assertTrue(all(np.array_equal(x[k],z[k]) for k in x))
    def test_compact_job_uses_original_outer_order_and_no_test_targets(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,t,code=fixture(d);attempt=root/'attempt';attempt.mkdir()
            frame,rows,classes,identity=M.prepare_job(root,p,t,Path(p['sources']['toy']['features'][t['config']]),attempt)
            self.assertTrue(np.array_equal(rows['outer'],np.flatnonzero(code!=2)))
            job=S.read(attempt/'job.json');self.assertFalse(job['test_outcomes_passed_to_worker']);self.assertEqual(job['adapter_source_sha256'],S.sha(M.WORKER))
            with np.load(attempt/'selection.npz') as z:self.assertEqual(set(z.files),{'outer','test'})
            with np.load(attempt/'state.npz') as z:self.assertEqual(len(z['y']),len(rows['outer']))
            self.assertEqual(identity['tokenizer_profile'],S.PROFILE)
    def test_gpu0_rejected_before_any_fit(self):
        with self.assertRaises(ValueError):M.fit('/tmp/no',{},dict(),0)

if __name__=='__main__':unittest.main()
