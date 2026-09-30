"""Small mechanism controls for reversal and unchanged spectators."""
from pathlib import Path
import json
import numpy as np
from drfp import DrfpEncoder
from synrfp import synrfp
from synrfp.graph.reaction import Reaction
from synrfp.tokenizers.wl import WLTokenizer
from synrfp.encode.core import fingerprint_from_tokens
from collections import Counter

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
BASE = "CCO>>CC=O"
CASES = {
    "base": BASE,
    "reversal": "CC=O>>CCO",
    "unchanged_reactant_spectator": "CCO.CCO>>CC=O.CCO",
    "unchanged_product_spectator": "CCO.CC=O>>CC=O.CC=O",
}


def syn_bits(r):
    return np.asarray(synrfp(r, tokenizer="wl", radius=2, sketch="parity",
                              bits=1024, seed=1, mode="delta",
                              node_attrs=["element", "aromatic", "charge", "hcount"],
                              edge_attrs=["order"], agents="both"), dtype=np.uint8)


def unsigned_delta_bits(r):
    parsed = Reaction.from_rsmi(r, agents="both")
    tok = WLTokenizer(node_attrs=["element", "aromatic", "charge", "hcount"],
                      edge_attrs=["order"])
    tokens_r = tok.tokens_graph(parsed.reactant, 2)
    tokens_p = tok.tokens_graph(parsed.product, 2)
    delta = Counter(tokens_p)
    for token, count in tokens_r.items():
        delta[token] -= count
        if delta[token] == 0:
            del delta[token]
    unsigned = Counter({token: abs(count) for token, count in delta.items()})
    # Place all unsigned support on the product side and use the union path;
    # this intentionally removes sign while preserving the changed-token set.
    return np.asarray(fingerprint_from_tokens(Counter(), unsigned, sketch="parity",
                                               bits=1024, seed=1, mode="union"), dtype=np.uint8)


def drfp_bits(r):
    return np.asarray(DrfpEncoder.encode(r, n_folded_length=2048, radius=3,
                                         rings=True, show_progress_bar=False)[0], dtype=np.uint8)


def main():
    base = {"synrfp": syn_bits(BASE), "synrfp_unsigned_delta": unsigned_delta_bits(BASE),
            "drfp": drfp_bits(BASE)}
    rows = []
    for case, reaction in CASES.items():
        for name, fn in [("synrfp", syn_bits), ("synrfp_unsigned_delta", unsigned_delta_bits),
                         ("drfp", drfp_bits)]:
            bits = fn(reaction)
            rows.append({"case": case, "reaction": reaction, "fingerprint": name,
                         "n_on_bits": int(bits.sum()),
                         "hamming_from_base": int(np.count_nonzero(bits != base[name])),
                         "identical_to_base": bool(np.array_equal(bits, base[name]))})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "representation_controls.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
