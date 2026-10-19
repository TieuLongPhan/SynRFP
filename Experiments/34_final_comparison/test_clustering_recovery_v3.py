import ast
import inspect
import json
import unittest
from unittest.mock import patch
import numpy as np
import clustering_recovery_v3 as R


class ValidationTests(unittest.TestCase):
    def test_exact_agreement_has_zero_discrepancy(self):
        X=np.eye(3,dtype=np.float32);centers=X.copy();native=np.arange(3)
        proof=R.numerical_oracle(X,centers,native,native)
        self.assertEqual(proof["mismatches"],0)
        self.assertTrue(proof["native_full_predict_equality_unchanged"])

    def test_near_coincident_float32_centers_accepted_with_explicit_float64_regret(self):
        X=np.ones((1,2048),np.float32)
        centers=np.concatenate([X.copy(),X.copy()]);centers[0,:]-=np.float32(1e-5)
        proof=R.numerical_oracle(X,centers,np.array([0]),np.array([1]))
        self.assertEqual(proof["mismatches"],1)
        self.assertGreater(proof["max_regret"],0)
        self.assertLess(proof["max_regret"],proof["max_forward_error_bound"])
        self.assertEqual(proof["disagreements"][0]["float64_best_label"],1)

    def test_materially_wrong_labels_or_centers_fail_bound(self):
        X=np.zeros((1,2048),np.float32)
        centers=np.stack([np.ones(2048,np.float32),np.zeros(2048,np.float32)])
        with self.assertRaisesRegex(ValueError,"exceeds float32"):
            R.numerical_oracle(X,centers,np.array([0]),np.array([1]))
        with self.assertRaisesRegex(ValueError,"oracle input"):
            R.numerical_oracle(X,centers,np.array([7]),np.array([1]))

    def test_bound_depends_on_expression_scale_and_dimension(self):
        X=np.ones((1,16),np.float32);centers=np.stack([X[0]-.00001,X[0]])
        a=R.numerical_oracle(X,centers,np.array([0]),np.array([1]))
        b=R.numerical_oracle(2*X,2*centers,np.array([0]),np.array([1]))
        self.assertAlmostEqual(b["max_forward_error_bound"]/a["max_forward_error_bound"],4)

    def test_ast_changes_only_secondary_comparison_keeps_native_equality(self):
        source=inspect.getsource(R.C.fit_cell)
        _,_,binding=R.adapted_fitter()
        self.assertEqual(binding["changed_ast_sites"],1)
        tree=ast.parse(source)
        native=ast.dump(ast.parse("np.array_equal(predicted, fitted)",mode="eval").body,include_attributes=False)
        self.assertEqual(sum(ast.dump(n,include_attributes=False)==native for n in ast.walk(tree)),1)
        self.assertIn("persisted_center",str(R.numerical_oracle(np.eye(2,dtype=np.float32),np.eye(2,dtype=np.float32),np.arange(2),np.arange(2))))

    def test_nonfinite_centers_and_changed_dimensions_refused(self):
        X=np.zeros((1,3),np.float32)
        for centers in (np.full((2,3),np.nan,np.float32),np.zeros((2,4),np.float32)):
            with self.assertRaises(ValueError):
                R.numerical_oracle(X,centers,np.array([0]),np.array([1]))

    def test_adapted_fitter_preserves_actual_native_training_weights_and_labels(self):
        import test_clustering as T
        fixture=T.ActualFitTests()
        fixture.setUp()
        try:
            R.C.fit_cell(fixture.task,fixture.plan)
            original=R.C.audit_cell(fixture.task,fixture.plan)
            centers=np.load(original["centers_path"]).copy()
            labels=np.load(original["assignment_path"])["labels"].copy()
            R.C.stem(fixture.task).with_suffix(".json").unlink()
            fn,proof,_=R.adapted_fitter()
            fn(fixture.task,fixture.plan)
            repaired=R.C.audit_cell(fixture.task,fixture.plan)
            np.testing.assert_array_equal(np.load(repaired["centers_path"]),centers)
            np.testing.assert_array_equal(np.load(repaired["assignment_path"])["labels"],labels)
            self.assertTrue(R.C.score_equal(original["scores"],repaired["scores"]))
            self.assertEqual(proof["oracle"]["rows"],6)
        finally:
            fixture.tearDown()

    def test_wrong_native_prediction_still_refused_before_completion(self):
        import test_clustering as T
        fixture=T.ActualFitTests()
        fixture.setUp()
        class WrongPredict(R.C.KMeans):
            def predict(self,X):
                return np.zeros(len(X),dtype=np.int32)
        try:
            with patch.object(R.C,"KMeans",WrongPredict):
                fn,_,_=R.adapted_fitter()
                with self.assertRaisesRegex(ValueError,"Saved center prediction replay differs"):
                    fn(fixture.task,fixture.plan)
            self.assertFalse(R.C.stem(fixture.task).with_suffix(".json").exists())
        finally:
            fixture.tearDown()

    def test_native_result_and_recovery_proof_are_written_atomically_together(self):
        import test_clustering as T
        fixture=T.ActualFitTests()
        fixture.setUp()
        try:
            root=fixture.root/"recovery"
            root.mkdir();(root/"plan.json").write_text("{}")
            native_write=R.C.write
            observed=[]
            def write(path,value):
                self.assertIn("validation_overlay",value)
                observed.append(value)
                native_write(path,value)
            with patch.object(R,"ROOT",root),patch.object(R.C,"write",side_effect=write):
                R.repaired_fit(fixture.task,fixture.plan,{"accepted_records_sha256":{}})
            self.assertEqual(len(observed),1)
            path=R.C.stem(fixture.task).with_suffix(".json")
            rec=json.loads(path.read_text())
            self.assertTrue(rec["validation_overlay"]["native_full_predict_vs_fit_exact"])
            self.assertEqual(rec["validation_overlay"]["independent_oracle"]["rows"],6)
        finally:
            fixture.tearDown()


if __name__=="__main__": unittest.main()
