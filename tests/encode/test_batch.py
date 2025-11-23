# tests/encode/test_batch.py
import unittest

import numpy as np

from synrfp.encode.batch import BatchEncoder

# Detect joblib for parallel tests
try:
    import joblib  # noqa: F401

    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False


class TestBatchEncoder(unittest.TestCase):
    def test_invalid_n_jobs(self):
        with self.assertRaises(ValueError):
            BatchEncoder(n_jobs=0)

    def test_invalid_bits_and_m_and_mode_and_batch_size(self):
        with self.assertRaises(ValueError):
            BatchEncoder(bits=0)
        with self.assertRaises(ValueError):
            BatchEncoder(m=0)
        with self.assertRaises(ValueError):
            BatchEncoder(mode="invalid")
        with self.assertRaises(ValueError):
            BatchEncoder(batch_size=0)
        with self.assertRaises(ValueError):
            BatchEncoder(batch_size=-5)

    def test_repr_contains_config(self):
        enc = BatchEncoder(tokenizer="wl", sketch="parity", bits=32, m=16)
        r = repr(enc)
        self.assertIn("BatchEncoder", r)
        self.assertIn("tokenizer='wl'", r)
        self.assertIn("sketch='parity'", r)

    def test_encode_one_and_many_wl_parity(self):
        enc = BatchEncoder(
            tokenizer="wl",
            sketch="parity",
            bits=32,
            m=16,
            mode="delta",
            n_jobs=1,
        )
        rsmi = "CCO>>CCO"
        fp1 = enc.encode_one(rsmi)
        self.assertIsInstance(fp1, np.ndarray)
        self.assertEqual(fp1.shape, (32,))

        fps = enc.encode_many([rsmi, rsmi])
        self.assertIsInstance(fps, np.ndarray)
        self.assertEqual(fps.shape, (2, 32))

    def test_classmethod_encode(self):
        rsmi_list = ["CCO>>CCO", "CCO>>CCO"]
        X = BatchEncoder.encode(
            rsmi_list,
            tokenizer="wl",
            sketch="parity",
            bits=16,
            m=8,
            mode="delta",
        )
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape, (2, 16))

    def test_batch_size_chunks_correctly(self):
        # batch_size=1 forces per-example chunks; behaviour should still match
        rsmi_list = ["CCO>>CCO", "CCO>>CCO", "CCO>>CCO"]
        enc = BatchEncoder(
            tokenizer="wl",
            sketch="parity",
            bits=32,
            m=16,
            mode="delta",
            n_jobs=1,
            batch_size=1,
        )
        X = enc.encode_many(rsmi_list)
        self.assertEqual(X.shape, (3, 32))

        # batch_size larger than dataset: single batch
        enc2 = BatchEncoder(
            tokenizer="wl",
            sketch="parity",
            bits=32,
            m=16,
            mode="delta",
            n_jobs=1,
            batch_size=10,
        )
        X2 = enc2.encode_many(rsmi_list)
        self.assertEqual(X2.shape, (3, 32))
        # results should be identical for same config
        np.testing.assert_array_equal(X, X2)

    @unittest.skipUnless(HAVE_JOBLIB, "joblib not available for parallel test")
    def test_parallel_encoding(self):
        enc = BatchEncoder(
            tokenizer="wl",
            sketch="parity",
            bits=32,
            m=16,
            mode="delta",
            n_jobs=2,
            batch_size=2,
        )
        rsmi_list = ["CCO>>CCO", "CCO>>CCO", "CCO>>CCO"]
        X = enc.encode_many(rsmi_list)
        self.assertEqual(X.shape, (3, 32))


if __name__ == "__main__":
    unittest.main()
