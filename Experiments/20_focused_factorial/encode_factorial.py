"""Label-free 72-cell SynRFP encoder with two frozen parser profiles.

The primary CW cells sketch unit signed/unsigned support AFTER full multiset
algebra. They are a new controlled sketch comparison, not native count-weighted
CW results. All vectors contain 2048 total bits; composite modes use two
1024-bit blocks. This module writes no caches and never reads targets.
"""
from __future__ import annotations

from collections import Counter
from numbers import Integral
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
for directory in (EXPERIMENTS, REPO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from synrfp.encode.core import fingerprint_from_tokens
from synrfp.graph.reaction import Reaction
from synrfp_exp.features import _tokenizer

NODE_ATTRS = ["element", "aromatic", "charge", "hcount"]
EDGE_ATTRS = ["order"]
TOKENIZERS = ("wl", "morgan", "path")
RADII = (2, 3)
SKETCHES = ("parity", "minhash", "cw")
MODES = ("delta", "union", "delta_union", "roles")
FP_SEED = 1
BBIT = 2
AGENTS = "both"
SCFG = {"node_attrs": NODE_ATTRS, "edge_attrs": EDGE_ATTRS,
        "agents": AGENTS, "bbit": BBIT}


def variant_name(tokenizer, radius, sketch, mode):
    return f"synrfp-{tokenizer}-r{radius}-b2048-{sketch}-{mode}"


VARIANTS = {
    variant_name(tokenizer, radius, sketch, mode): {
        "tokenizer": tokenizer, "radius": radius, "sketch": sketch,
        "mode": mode, "n_bits": 2048, "kind": "bits", "seed": FP_SEED,
        "bbit": BBIT,
        "count_policy": ("unit_support_after_full_multiset_algebra" if sketch == "cw"
                         else "native_unweighted_support_after_full_multiset_algebra"),
        **({"block_bits": [1024, 1024], "block_order": ["signed_delta", "union"]}
           if mode == "delta_union" else {}),
        **({"block_bits": [1024, 1024], "block_order": ["signed_delta", "shared"]}
           if mode == "roles" else {}),
    }
    for tokenizer in TOKENIZERS for radius in RADII
    for sketch in SKETCHES for mode in MODES
}

PROFILES = {
    "classification": {
        "parser": "synrfp.graph.reaction.Reaction.from_rsmi",
        "agents": AGENTS,
        "boundaries": "native agent placement, name truncation and dative-arrow boundary retained",
    },
    "yield": {
        "parser": "13_yield.bh_features._strict_molecule + _graph + Reaction.from_graph",
        "delimiter": "exactly one >>; explicit molecular sides preserve embedded -> bonds",
        "agents": "source explicit sides; no middle-field redistribution",
    },
}
PROTOCOL = {
    "schema": "synrfp_focused_72_cells_unit_support_v1", "variants": 72,
    "tokenizers": list(TOKENIZERS), "radii": list(RADII), "sketches": list(SKETCHES),
    "modes": list(MODES), "total_bits": 2048, "seed": FP_SEED, "bbit": BBIT,
    "node_attrs": NODE_ATTRS, "edge_attrs": EDGE_ATTRS, "profiles": PROFILES,
    "delta": "D=P-R from full occurrence counts before support reduction",
    "union": "U=P+R from full occurrence counts before support reduction",
    "shared": "S=min(R,P), original token IDs emitted once if positive",
    "cw_policy": "new primary unit-support CW; signed D retained, |D|/U/S weights set to one only after algebra",
    "native_weighted_cw": "not part of this primary 72-cell comparison",
    "role_meaning": "shared local token environments; not mapped-atom retention",
    "invalid_rows": "all 72 zero rows, common ok=False; no row removal",
    "construction_errors": "raise; never relabel feature defects as parse failures",
    "stereo_features": False, "labels_read": False,
}


def token_algebra(tokens_r, tokens_p):
    """Return complete signed D, unsigned U and S; preserve partial sharing."""
    for counts in (tokens_r, tokens_p):
        if any(isinstance(count, bool) or not isinstance(count, Integral) or count < 0
               for count in counts.values()):
            raise ValueError("Token counts must be nonnegative integers")
    delta, union, shared = Counter(), Counter(), Counter()
    for token in tokens_r.keys() | tokens_p.keys():
        r, p = int(tokens_r.get(token, 0)), int(tokens_p.get(token, 0))
        if p != r:
            delta[token] = p - r
        if p + r:
            union[token] = p + r
        if min(r, p):
            shared[token] = min(r, p)
    return delta, union, shared


def _packed(tokens_r, tokens_p, sketch, bits, mode):
    raw = fingerprint_from_tokens(tokens_r, tokens_p, sketch=sketch, bits=bits,
                                  seed=FP_SEED, mode=mode, bbit=BBIT)
    if (raw.dtype != np.uint8 or raw.shape != (bits,) or
            not np.logical_or(raw == 0, raw == 1).all()):
        raise ValueError("Malformed native binary fingerprint")
    return np.packbits(raw)


def encode_tokens(tokens_r, tokens_p, sketch):
    """Encode four modes using full-count algebra and declared sketch weights."""
    if sketch not in SKETCHES:
        raise ValueError("Unknown primary sketch")
    delta, union, shared = token_algebra(tokens_r, tokens_p)
    if sketch == "cw":
        # Reduce only AFTER the signed difference; (R=2,P=1) must remain lost.
        delta_r = Counter({token: 1 for token, count in delta.items() if count < 0})
        delta_p = Counter({token: 1 for token, count in delta.items() if count > 0})
        union_r, union_p = Counter(), Counter({token: 1 for token in union})
    else:
        delta_r, delta_p = tokens_r, tokens_p
        union_r, union_p = tokens_r, tokens_p
    d1024 = _packed(delta_r, delta_p, sketch, 1024, "delta")
    u1024 = _packed(union_r, union_p, sketch, 1024, "union")
    # Unsigned keys match direct ParityFold(shared.keys()) byte-for-byte.
    s1024 = _packed(Counter(), Counter({token: 1 for token in shared}), sketch, 1024, "union")
    return {
        "delta": _packed(delta_r, delta_p, sketch, 2048, "delta"),
        "union": _packed(union_r, union_p, sketch, 2048, "union"),
        "delta_union": np.concatenate((d1024, u1024)),
        "roles": np.concatenate((d1024, s1024)),
    }


def _reaction(rsmi, profile):
    if profile == "classification":
        return Reaction.from_rsmi(rsmi, agents=AGENTS)
    if not isinstance(rsmi, str) or rsmi.count(">>") != 1:
        raise ValueError("Yield inputs require exactly one >> between explicit molecular sides")
    left, right = rsmi.split(">>")
    directory = EXPERIMENTS / "13_yield"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import bh_features as B
    B._strict_molecule(left)
    B._strict_molecule(right)
    return Reaction.from_graph(B._graph(left), B._graph(right))


def _tokens(rsmi, profile, tokenizers):
    reaction = _reaction(rsmi, profile)
    return {(name, radius): (tokenizer.tokens_graph(reaction.reactant, radius),
                            tokenizer.tokens_graph(reaction.product, radius))
            for name, tokenizer in tokenizers.items() for radius in RADII}


def _from_tokens(tokens):
    result = {}
    for (tokenizer, radius), (reactant, product) in tokens.items():
        for sketch in SKETCHES:
            for mode, value in encode_tokens(reactant, product, sketch).items():
                if value.dtype != np.uint8 or value.shape != (256,):
                    raise ValueError("Malformed fixed-width feature")
                result[variant_name(tokenizer, radius, sketch, mode)] = value
    if set(result) != set(VARIANTS):
        raise ValueError("Incomplete primary factorial feature population")
    return result


def encode_reactions(strings, profile="classification"):
    """Return 72 packed arrays and common validity in exactly input order.

    Parse/tokenization failures preserve zero rows. Construction errors raise.
    Genuine source IDs and source positions are the caller's responsibility.
    """
    if profile not in PROFILES:
        raise ValueError("Choose classification or yield parser profile")
    if isinstance(strings, (str, bytes)):
        raise TypeError("Expected a collection of reaction strings")
    strings = list(strings)
    arrays = {name: np.zeros((len(strings), 256), dtype=np.uint8) for name in VARIANTS}
    ok = np.zeros(len(strings), dtype=bool)
    tokenizers = {name: _tokenizer(name, NODE_ATTRS, EDGE_ATTRS) for name in TOKENIZERS}
    for index, rsmi in enumerate(strings):
        try:
            tokens = _tokens(rsmi, profile, tokenizers)
        except Exception:
            continue
        values = _from_tokens(tokens)
        for name in VARIANTS:
            arrays[name][index] = values[name]
        ok[index] = True
    return arrays, ok
