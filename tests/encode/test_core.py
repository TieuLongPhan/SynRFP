# tests/encode/test_core.py
import unittest

from synrfp.encode.core import synrfp

# Optional tokenizers / sketchers for smoke tests
try:
    from synrfp.tokenizers.morgan import MorganTokenizer  # noqa: F401

    HAVE_MORGAN = True
except Exception:
    HAVE_MORGAN = False

try:
    from synrfp.tokenizers.path import PathTokenizer  # noqa: F401

    HAVE_PATH = True
except Exception:
    HAVE_PATH = False

try:
    from synrfp.sketchers.srp_sketch import SRPSketch  # noqa: F401

    HAVE_SRP = True
except Exception:
    HAVE_SRP = False


class TestSynRFP(unittest.TestCase):
    def test_wl_parity_delta_basic_all_zero_for_identical_sides(self):
        """Delta mode on identical reactant/product should give an all-zero bit vector."""
        rsmi = "CCO>>CCO"
        bits = 64
        fp = synrfp(
            rsmi, tokenizer="wl", sketch="parity", bits=bits, mode="delta", seed=0
        )
        # type & length
        self.assertIsInstance(fp, list)
        self.assertEqual(len(fp), bits)
        # binary
        self.assertTrue(all(b in (0, 1) for b in fp))
        # identical sides -> no net change -> all zeros expected for delta
        self.assertTrue(
            all(b == 0 for b in fp),
            "Delta fingerprint for identical sides should be all zeros",
        )

    def test_wl_parity_union_nonzero_for_identical_sides(self):
        """
        Union mode on identical sides should include tokens present on either side
        and therefore produce a non-empty fingerprint (at least one bit set).
        """
        rsmi = "CCO>>CCO"
        bits = 64
        fp = synrfp(
            rsmi, tokenizer="wl", sketch="parity", bits=bits, mode="union", seed=0
        )
        self.assertEqual(len(fp), bits)
        # Expect at least one 1 bit for union (since there are atoms/fragments present)
        self.assertTrue(
            any(b == 1 for b in fp),
            "Union fingerprint for identical sides should have at least one bit set",
        )

    def test_deterministic_seed_reproducible_and_different_seed_changes(self):
        """Same seed should give identical fingerprints; different seed very likely different."""
        rsmi = "CCO.O>>CC(=O)O"  # a simple reaction with a change
        bits = 128
        fp1 = synrfp(
            rsmi, tokenizer="wl", sketch="parity", bits=bits, seed=42, mode="delta"
        )
        fp2 = synrfp(
            rsmi, tokenizer="wl", sketch="parity", bits=bits, seed=42, mode="delta"
        )
        self.assertEqual(
            fp1, fp2, "Fingerprints with identical seeds must be identical"
        )

        # different seed -> fingerprint should (very likely) differ
        fp3 = synrfp(
            rsmi, tokenizer="wl", sketch="parity", bits=bits, seed=123, mode="delta"
        )
        # It's astronomically unlikely they match exactly for different seeds; assert inequality.
        self.assertNotEqual(
            fp1, fp3, "Fingerprints with different seeds should usually differ"
        )

    def test_invalid_tokenizer_raises(self):
        with self.assertRaises(ValueError):
            synrfp("CCO>>CCO", tokenizer="nonexistent")

    def test_invalid_sketch_raises(self):
        with self.assertRaises(ValueError):
            synrfp("CCO>>CCO", sketch="nonexistent")

    def test_bits_must_be_positive_int(self):
        """Invalid bits (zero or negative) should raise ValueError when using parity sketch."""
        with self.assertRaises(ValueError):
            synrfp("CCO>>CCO", tokenizer="wl", sketch="parity", bits=0)
        with self.assertRaises(ValueError):
            synrfp("CCO>>CCO", tokenizer="wl", sketch="parity", bits=-128)

    @unittest.skipUnless(HAVE_MORGAN, "MorganTokenizer not available")
    def test_morgan_parity_basic(self):
        rsmi = "CCO>>CCO"
        fp = synrfp(rsmi, tokenizer="morgan", sketch="parity", bits=32, seed=0)
        self.assertEqual(len(fp), 32)
        self.assertTrue(all(b in (0, 1) for b in fp))

    @unittest.skipUnless(HAVE_PATH, "PathTokenizer not available")
    def test_path_parity_basic(self):
        rsmi = "CCO>>CCO"
        fp = synrfp(rsmi, tokenizer="path", sketch="parity", bits=32, seed=0)
        self.assertEqual(len(fp), 32)
        self.assertTrue(all(b in (0, 1) for b in fp))

    @unittest.skipUnless(HAVE_SRP, "SRPSketch not available")
    def test_wl_srp_basic(self):
        rsmi = "CCO>>CCO"
        # SRP uses weighted path; signature_to_bits maps to `bits` length
        fp = synrfp(rsmi, tokenizer="wl", sketch="srp", bits=64, m=32, seed=0)
        self.assertEqual(len(fp), 64)
        self.assertTrue(all(b in (0, 1) for b in fp))

    def test_various_bit_lengths(self):
        """Check small and larger bit sizes for parity sketch return correct length."""
        rsmi = "CBr>>CBr"  # identical sides -> delta zeros, but length must match
        for bits in (8, 16, 32, 64, 128):
            fp = synrfp(
                rsmi, tokenizer="wl", sketch="parity", bits=bits, mode="delta", seed=0
            )
            self.assertEqual(len(fp), bits)
            self.assertTrue(all(b in (0, 1) for b in fp))

    def test_delta_vs_union_difference(self):
        """
        For a reaction that changes structure, delta and union should not be identical.
        This sanity-check ensures mode has semantic effect.
        """
        rsmi = "CCO.O>>CC(=O)O"
        bits = 64
        f_delta = synrfp(
            rsmi, tokenizer="wl", sketch="parity", bits=bits, mode="delta", seed=0
        )
        f_union = synrfp(
            rsmi, tokenizer="wl", sketch="parity", bits=bits, mode="union", seed=0
        )
        self.assertNotEqual(
            f_delta,
            f_union,
            "delta and union fingerprints should differ for non-trivial reactions",
        )


if __name__ == "__main__":
    unittest.main()
