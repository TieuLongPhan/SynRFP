import copy
import itertools
from pathlib import Path
import tempfile
import unittest
import study as S

def grid():
    configs={str(i):dict(tokenizer=t,radius=r,sketch=s,mode=m) for i,(t,r,s,m) in enumerate(itertools.product(('morgan',),(2,3),('parity','minhash','cw'),('delta','union','delta_union','roles')))}
    ds=list(S.DATASETS);cluster=list(S.CLUSTER_DATASETS)
    tasks={name:[dict(task=name,dataset=d,config=c,**{repeat:f}) for d in scope for c in configs for f in range(5)]
        for name,repeat,scope in [('classification','fold',ds),('clustering','seed',cluster)]}
    return dict(schema='synrfp-morgan-ecfp-correction-v1',tokenizer_profile=S.PROFILE,classification_datasets=ds,
        clustering_datasets=cluster,configs=configs,tasks=tasks,sources={str(i):{} for i in range(9)},row_cap=None,new_filtering=False)

class StudyTests(unittest.TestCase):
    def test_exact_grid(self):
        p=S.validate_grid(grid());self.assertEqual([len(p['tasks'][k]) for k in ('classification','clustering')],[1680,1440])
    def test_missing_or_duplicate_cell_rejected(self):
        for kind in ('classification','clustering'):
            p=grid();p['tasks'][kind][-1]=p['tasks'][kind][0]
            with self.assertRaises(ValueError):S.validate_grid(p)
    def test_ehreact_rejected(self):
        p=grid();p['classification_datasets'][-1]='ehreact_x'
        with self.assertRaises(ValueError):S.validate_grid(p)
    def test_wrong_twelve_endpoint_scope_rejected_even_with_complete_crossing(self):
        p=grid();p['clustering_datasets'][-1]='syntemp_r1'
        for task in p['tasks']['clustering']:
            if task['dataset']=='mech_2':task['dataset']='syntemp_r1'
        with self.assertRaises(ValueError):S.validate_grid(p)
    def test_old_morgan_or_filter_rejected(self):
        for key,value in [('tokenizer_profile','old_wl_like_morgan'),('row_cap',20000),('new_filtering',True)]:
            p=grid();p[key]=value
            with self.assertRaises(ValueError):S.validate_grid(p)
    def test_hash_cache_detects_same_length_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_text('abc');h=S.sha(p);c=S.HashCache();c.verify(p,h);p.write_text('def')
            with self.assertRaises(ValueError):c.verify(p,h)
    def test_result_names_distinguish_tasks_seeds(self):
        a=dict(task='classification',dataset='x',config='m',fold=0);b=dict(task='clustering',dataset='x',config='m',seed=0)
        self.assertNotEqual(S.task_id(a),S.task_id(b));self.assertIn('/artifacts/clustering/',str(S.result_path('/tmp/z',b)))

if __name__=='__main__':unittest.main()
