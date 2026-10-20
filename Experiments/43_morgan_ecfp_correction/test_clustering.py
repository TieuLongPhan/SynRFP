import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.cluster import MiniBatchKMeans
import study as S
import clustering as C
from test_models import fixture

class ClusterTests(unittest.TestCase):
    def test_raw_binary_original_math_and_full_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,t,_=fixture(d);t={**t,'task':'clustering','seed':0};t.pop('fold')
            with patch.object(C.E,'feature_ready',return_value={'phase':'complete'}),patch.object(C.E,'disk_guard'):
                f,X,ok,fp=C.data(p,t);self.assertEqual(set(np.unique(X)),{0.,1.})
                with threadpool_limits(limits=1):
                    oracle=MiniBatchKMeans(n_clusters=4,random_state=0,**p['clustering_params']);expected=oracle.fit_predict(X)
                C.fit(root,p,t,1);r=C.verify_result(root,p,t,deep=True)
                with np.load(S.result_path(root,t).with_suffix('.npz')) as z:self.assertTrue(np.array_equal(z['labels'],expected))
                self.assertFalse(r['labels_used_for_fit_or_selection']);self.assertEqual(r['n'],40)
                bad={**r,'ami':r['ami']+.1}
                with self.assertRaises(ValueError):C.verify_result(root,p,t,bad,deep=True)
    def test_row_identity_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,t,_=fixture(d);t={**t,'task':'clustering','seed':0};t.pop('fold')
            with patch.object(C.E,'feature_ready',return_value={'phase':'complete'}),patch.object(C.E,'disk_guard'):
                C.fit(root,p,t,1);r=S.read(S.result_path(root,t));a=S.result_path(root,t).with_suffix('.npz')
                with np.load(a) as z:values={k:z[k] for k in z.files}
                values['source_positions']=values['source_positions'][::-1];S.npz(a,**values)
                r['assignments_sha256']=S.sha(a)
                with self.assertRaises(ValueError):C.verify_result(root,p,t,r)

if __name__=='__main__':unittest.main()
