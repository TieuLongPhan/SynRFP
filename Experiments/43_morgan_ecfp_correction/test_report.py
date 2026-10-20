import itertools
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import report as R
import study as S

def fixture(task='classification'):
    a,_=R.analysis();configs={f'{t}-{r}-{s}-{m}':dict(tokenizer=t,radius=r,sketch=s,mode=m) for t,r,s,m in itertools.product(*a.LEVELS.values())}
    corrected={k:v for k,v in configs.items() if v['tokenizer']=='morgan'}
    plan=dict(classification_datasets=['toy'],clustering_datasets=['toy'],configs=corrected,tokenizer_profile='synrfp-ecfp-count-v1')
    rows=[];key='fold' if task=='classification' else 'seed'
    for config,spec in configs.items():
        for seed in range(5):
            radius=spec['radius']-2;union=spec['mode']=='union';score=.1+radius*.02+union*.03+radius*union*.04+seed*.001
            rows.append(dict(dataset='toy',config=config,**{key:seed},mcc=score,accuracy=score+.1,macro_f1=score+.05,
                ami=score,ari=score,nmi=score,silhouette=score,largest_cluster_fraction=score,reaction_singleton_fraction=score,actual_clusters=10,fit_seconds=1))
    old=pd.DataFrame(rows);new=old[old.config.isin(corrected)].copy();new['mcc']+=.01
    return plan,dict(configs=configs),old,new

class ReportTests(unittest.TestCase):
    def test_exact48plus24_and_no_old_morgan(self):
        p,o,old,new=fixture();combined=R.combine(old,new,p,o,'classification')
        self.assertEqual(len(combined),360);self.assertEqual(combined.result_origin.value_counts().to_dict(),{'reused-study22-WL-Path':240,'fresh-study43-ECFP':120})
        merged=combined[combined.tokenizer.eq('morgan')].merge(new,on=['dataset','config','fold'],suffixes=('_actual','_expected'))
        self.assertTrue(np.array_equal(merged.mcc_actual,merged.mcc_expected))
    def test_missing_corrected_or_duplicated_control_refused(self):
        p,o,old,new=fixture()
        for controls,fresh in ((old,new.iloc[:-1]),(pd.concat([old,old.iloc[:1]]),new)):
            with self.assertRaises(ValueError):R.combine(controls,fresh,p,o,'classification')
    def test_legacy_morgan_extremes_cannot_leak(self):
        p,o,old,new=fixture();old.loc[old.config.isin(p['configs']),'mcc']=999
        frame=R.combine(old,new,p,o,'classification');self.assertLess(frame.mcc.max(),1)
    def test_wrong_fresh_profile_scope_refused(self):
        p,o,old,new=fixture();new.loc[new.index[0],'config']=next(k for k,v in o['configs'].items() if v['tokenizer']=='wl')
        with self.assertRaises(ValueError):R.combine(old,new,p,o,'classification')
    def test_all_four_orders_and_known_radius_mode_interaction(self):
        p,o,old,new=fixture();frame=R.combine(old,new,p,o,'classification');tables,refusals=R.tables(frame,p,o,'classification')
        effects=tables['paired_full_interactions.csv'];self.assertEqual(set(effects.order),{1,2,3,4});self.assertFalse(refusals)
        row=effects[(effects.metric=='mcc')&(effects.effect=='radius:mode')&(effects.level=='3:union')]
        self.assertTrue(np.allclose(row.contrast,.04,atol=1e-12));self.assertTrue(tables['cross_panel_consistency.csv'].diagnostic_only_not_config_selection.all())
    def test_nonfinite_secondary_refuses_affected_contrast_not_primary(self):
        p,o,old,new=fixture('clustering');new.loc[new.index[0],'silhouette']=np.nan
        frame=R.combine(old,new,p,o,'clustering');tables,refusals=R.tables(frame,p,o,'clustering')
        self.assertEqual(len(refusals),1);self.assertEqual(refusals[0]['metric'],'silhouette')
        self.assertIn('ami',set(tables['paired_full_interactions.csv'].metric))
    def test_original_native_digest_catalog_is_authoritative(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);catalog=folder/'inputs.json';S.write(catalog,{'/retained/result.json':'a'*64})
            receipt=folder/'COMPLETE.json';S.write(receipt,dict(phase='complete',canonical_metric_and_artifact_checks_passed=True,
                plan_sha256='b'*64,outputs_sha256={'inputs.json':S.sha(catalog)}))
            p=dict(old_wl_path_controls={'classification':{'path':str(folder/'cells.csv')}},source_plan_sha256='b'*64,
                pinned_files={str(receipt.resolve()):S.sha(receipt)})
            self.assertEqual(R.native_catalog(S,p,'classification',{},S.HashCache())['/retained/result.json'],'a'*64)
            S.write(catalog,{'/retained/result.json':'c'*64})
            with self.assertRaises(ValueError):R.native_catalog(S,p,'classification',{},S.HashCache())
    def test_corrected_worker_keyed_catalog_must_be_exact(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);S.write(root/'plan.json',{})
            tasks=[dict(task='classification',dataset='toy',config='m',fold=f) for f in range(2)]
            p=dict(tasks={'classification':tasks},sources_sha256={})
            outputs={str(S.result_path(root,t)):'a'*64 for t in tasks}
            receipt=dict(phase='complete',stage='mlp',worker=0,workers=1,plan_sha256=S.sha(root/'plan.json'),all_shard_items_verified=True,
                audited_items=2,expected_items=2,sources_sha256={},outputs_sha256=outputs)
            path=root/'mlp_worker_0_COMPLETE.json';S.write(path,receipt)
            self.assertEqual(len(R.stage_catalog(S,root,p,'classification',{},S.HashCache())),2)
            receipt['outputs_sha256']={next(iter(outputs)):'a'*64};S.write(path,receipt)
            with self.assertRaises(ValueError):R.stage_catalog(S,root,p,'classification',{},S.HashCache())

if __name__=='__main__':unittest.main()
