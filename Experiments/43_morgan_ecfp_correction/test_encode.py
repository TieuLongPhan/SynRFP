from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
import encode as E
import study as S

class EncodeTests(unittest.TestCase):
    def test_foreign_loaded_core_rejected(self):
        with patch.dict(sys.modules,{'synrfp.foreign':SimpleNamespace(__file__='/tmp/wrong/synrfp/a.py')}):
            with self.assertRaises(ValueError):E.loaded_core_is_correct(S.CORRECTED)
    def test_correct_worktree_module_accepted(self):
        with patch.dict(sys.modules,{'synrfp.test_profile':SimpleNamespace(__file__=str(S.CORRECTED/'synrfp/a.py'))}):
            E.loaded_core_is_correct(S.CORRECTED)
    def test_wrong_shape_and_construction_failure_are_fatal(self):
        fake=SimpleNamespace(VARIANTS={'x':dict(tokenizer='morgan')},_reaction=lambda *a:(_ for _ in ()).throw(RuntimeError('parse defect')))
        with patch.object(E,'engine',return_value=(fake,None)):
            with self.assertRaisesRegex(RuntimeError,'parse defect'):E.encode_reactions(['bad'],[True])
            result=E.encode_reactions(['bad'],[False]);self.assertFalse(result['x'].any())
    def test_row_mask_length_mismatch(self):
        with patch.object(E,'engine',return_value=(None,None)):
            with self.assertRaises(ValueError):E.encode_reactions(['x'],[])
    def test_only_closed_owned_staging_compacted_and_resume_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);S.write(root/'plan.json',{});receipt=root/'artifacts/features/s/COMPLETE.json';S.write(receipt,{'phase':'complete'})
            directory=root/'artifacts/encoding_journals/s';directory.mkdir(parents=True)
            p=directory/'0.npz';S.npz(p,x=np.arange(2));S.write(p.with_suffix('.json'),{'a':1});unrelated=directory/'unrelated.npz';unrelated.write_bytes(b'keep')
            plan={'sources':{'s':{'feature_receipt':str(receipt)}}}
            S.write(directory/'CHUNKS_SEALED.json',dict(plan_sha256=S.sha(root/'plan.json'),source='s',tokenizer_profile=S.PROFILE,
                chunks={p.name:dict(sha256=S.sha(p),receipt_sha256=S.sha(p.with_suffix('.json')))}))
            with patch.object(E,'feature_ready',return_value={'phase':'complete'}):
                self.assertEqual(E.compact_staging(root,plan,'s'),1);self.assertFalse(p.exists());self.assertTrue(unrelated.exists())
                self.assertEqual(E.compact_staging(root,plan,'s'),0)
                with patch.object(E,'feature_ready',return_value=None):
                    with self.assertRaises(ValueError):E.compact_staging(root,plan,'s')
    def test_symlink_staging_never_removed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);S.write(root/'plan.json',{});receipt=root/'COMPLETE.json';S.write(receipt,{'phase':'complete'})
            directory=root/'artifacts/encoding_journals/s';directory.mkdir(parents=True);target=root/'valuable.npz';target.write_bytes(b'keep')
            p=directory/'0.npz';p.symlink_to(target);S.write(p.with_suffix('.json'),{})
            S.write(directory/'CHUNKS_SEALED.json',dict(plan_sha256=S.sha(root/'plan.json'),source='s',tokenizer_profile=S.PROFILE,
                chunks={p.name:dict(sha256=S.sha(p),receipt_sha256=S.sha(p.with_suffix('.json')))}))
            with patch.object(E,'feature_ready',return_value={'phase':'complete'}):
                with self.assertRaises(ValueError):E.compact_staging(root,{'sources':{'s':{'feature_receipt':str(receipt)}}},'s')
            self.assertEqual(target.read_bytes(),b'keep')

if __name__=='__main__':unittest.main()
