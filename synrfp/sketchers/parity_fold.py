# -----------------------------------------------------------------------------
# synrfp/sketchers/parity_fold.py
from typing import Iterable
from synrfp.sketchers.base import BaseSketch


class ParityFold(BaseSketch):
    """
    **ParityFold** — XOR-fold a set of tokens into a fixed-length bit array.

    :param bits: Length of the bit array.
    :type bits: int
    :param seed: Random seed influencing the hashing of indices.
    :type seed: int

    Each token hashes to ``idx = hash((seed, token)) % bits``;
    the bit at *idx* is toggled.
    The resulting :pyclass:`bytearray` contains 0/1 integers (one per bit).
    """

    def __init__(self, bits: int = 1024, seed: int = 1):
        if bits <= 0:
            raise ValueError("bits must be positive")
        super().__init__(seed)
        self.bits = int(bits)

    def __repr__(self) -> str:  # noqa: D401
        return f"ParityFold(bits={self.bits}, seed={self.seed})"

    def build(self, support: Iterable[int]) -> bytearray:  # noqa: D401
        bits = bytearray(self.bits)
        for h in support:
            idx = hash((self.seed, int(h))) % self.bits
            bits[idx] ^= 1
        return bits

    def describe(self) -> str:  # noqa: D401
        return (
            "# Binary parity sketch (1024 bits)\n"
            "pf = ParityFold(bits=1024, seed=42)\n"
            "sk = pf.build(tokens)\n"
        )
