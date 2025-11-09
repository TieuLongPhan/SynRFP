# ----------------------------------------------------------------------------
# tests/test_encoder.py
import unittest
import numpy as np

from synrfp.encoder import SynRFPEncoder


class TestSynRFPEncoder(unittest.TestCase):

    def test_single_encode_default(self):
        """Single reaction, default parameters (WL + parity)."""
        rxn_smiles = ["CCO>>C=C.O"]
        fps = SynRFPEncoder.encode(rxn_smiles, bits=16, seed=0)
        self.assertIsInstance(fps, np.ndarray)
        self.assertEqual(fps.shape, (1, 16))
        # bits should be 0 or 1
        self.assertTrue(np.all((fps == 0) | (fps == 1)))

    def test_batch_encode(self):
        """Multiple reactions produce an array with correct first dimension."""
        rxn_smiles = ["CCO>>C=C.O", "CC>>C.C"]
        fps = SynRFPEncoder.encode(rxn_smiles, bits=8, seed=1)
        self.assertIsInstance(fps, np.ndarray)
        self.assertEqual(fps.shape, (2, 8))
        # ensure each row is 0/1
        self.assertTrue(np.all((fps == 0) | (fps == 1)))

    def test_invalid_tokenizer(self):
        """Unknown tokenizer name should raise ValueError."""
        with self.assertRaises(ValueError):
            SynRFPEncoder.encode(["CCO>>C=C.O"], tokenizer="foobar")

    def test_invalid_sketch(self):
        """Unknown sketch name should raise ValueError."""
        with self.assertRaises(ValueError):
            SynRFPEncoder.encode(["CCO>>C=C.O"], sketch="foobar")

    def test_inconsistent_lengths_error(self):
        """
        If underlying synrfp returns vectors of differing lengths,
        encode() should raise a ValueError.
        """
        import synrfp.encoder as enc_mod

        # Monkey-patch synrfp to produce different lengths
        original = enc_mod.synrfp

        def fake_synrfp(rsmi, **kwargs):
            return [0, 1] if rsmi == "A" else [1, 0, 1]

        enc_mod.synrfp = fake_synrfp
        try:
            with self.assertRaises(ValueError):
                SynRFPEncoder.encode(["A", "B"])
        finally:
            # Restore original function
            enc_mod.synrfp = original


if __name__ == "__main__":
    unittest.main()
