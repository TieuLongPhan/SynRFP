"""Independent native preprocessing and mutation controls for final regressors."""
import tempfile
import unittest
import os
from pathlib import Path

import numpy as np
import regression as R


class RegressionTests(unittest.TestCase):
    def test_native_preprocessing_against_independent_oracle(self):
        rng=np.random.RandomState(7)
        binary=rng.randint(0,2,(60,2048)).astype(np.uint8)
        labels=np.arange(60,dtype=np.float64)**2
        rows=(np.arange(30),np.arange(30,45),np.arange(45,60))
        fit,test,stats=R.training_arrays(np.packbits(binary,axis=1),labels,rows)
        mean,scale=labels[:30].mean(),labels[:30].std()
        np.testing.assert_array_equal(fit['X_train'],binary[:30].astype(np.float32))
        np.testing.assert_array_equal(fit['X_val'],binary[30:45].astype(np.float32))
        np.testing.assert_array_equal(test,binary[45:].astype(np.float32))
        np.testing.assert_array_equal(fit['y_train'],((labels[:30]-mean)/scale).astype(np.float32))
        np.testing.assert_array_equal(fit['y_val'],((labels[30:45]-mean)/scale).astype(np.float32))
        self.assertEqual(float(stats['target_mean']),mean)
        self.assertEqual(float(stats['target_scale']),scale)
        self.assertNotIn('y_test',fit)

    def test_heldout_outcomes_do_not_change_inputs(self):
        x=np.zeros((60,256),np.uint8);y=np.arange(60,dtype=np.float64)
        rows=(np.arange(30),np.arange(30,45),np.arange(45,60))
        a=R.training_arrays(x,y,rows)
        y[45:]=-100000
        b=R.training_arrays(x,y,rows)
        for first,second in zip(a,b):
            if isinstance(first,dict):
                for k in first:np.testing.assert_array_equal(first[k],second[k])
            else:np.testing.assert_array_equal(first,second)

    def test_fraction_respects_frozen_randomstate_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'split.npz'
            np.savez(p,train=np.arange(100),validation=np.arange(100,115),test=np.arange(115,130))
            rows=R.selectors(p,np.ones(130,bool),.25)
            np.testing.assert_array_equal(rows[0],np.random.RandomState(42).permutation(100)[:25])
            np.testing.assert_array_equal(rows[1],np.arange(100,115))

    def test_injected_outcomes_and_feature_mutation_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fit.npz';expected={'X_train':np.zeros((2,3),np.float32)}
            np.savez(p,**expected,y_test=np.array([1]))
            with self.assertRaises(ValueError):R.require_arrays(p,expected)
            np.savez(p,X_train=np.ones((2,3),np.float32))
            with self.assertRaises(ValueError):R.require_arrays(p,expected)
            np.savez(p,X_train=np.zeros((2,3),np.float64))
            with self.assertRaises(ValueError):R.require_arrays(p,expected)

    def test_overlapping_split_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'split.npz';np.savez(p,train=np.arange(20),validation=np.arange(10,30),test=np.arange(30,45))
            with self.assertRaises(ValueError):R.selectors(p,np.ones(45,bool),1.)

    def test_metric_replay_and_nonfinite_refusal(self):
        self.assertEqual(R.metrics([1.,2.,3.],[1.,2.,3.]),dict(mae=0.,rmse=0.,r2=1.))
        self.assertAlmostEqual(R.metrics([1.,2.,3.],[2.,3.,4.])['r2'],-.5)
        with self.assertRaises(ValueError):R.metrics([1.,2.],[1.,float('nan')])

    def test_checksum_cache_cannot_hide_same_size_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_bytes(b'123');before=R.sha(p)
            p.write_bytes(b'321');self.assertNotEqual(before,R.sha(p))

    def test_repeated_checksum_hits_and_restored_mtime_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_bytes(b'abc');original=p.stat();first=R.sha(p)
            hits=R._HASH_HITS
            for _ in range(20):self.assertEqual(first,R.sha(p))
            self.assertEqual(R._HASH_HITS-hits,20)
            p.write_bytes(b'def')
            os.utime(p,ns=(original.st_atime_ns,original.st_mtime_ns))
            self.assertNotEqual(first,R.sha(p))


if __name__=='__main__':unittest.main()
