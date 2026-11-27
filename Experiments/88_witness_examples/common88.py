"""Read the frozen source, count, and fingerprint receipts used for witnesses."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
BRANCH75 = WORKSPACE / "SynRFP-chemistry-cases/Experiments/75_chemistry_cases"
BRANCH74 = WORKSPACE / "SynRFP-delta-ceiling/Experiments/74_delta_ceiling"
sys.path[:0] = [str(BRANCH75), str(BRANCH74)]

from common import (
    BASE, CEILING, STUDY22, InputAudit, chemistry, feature, load_plan, panel_data,
    read, source_data, write,
)
from drfp_cases import rooted_examples
import encode_counts as counts

ROOT = BASE / "request88_witness_examples_v1"
OUTPUT = HERE / "publication_metadata/witness_examples_v1"
TOKENIZER = counts.F._tokenizer("wl", counts.F.NODE_ATTRS, counts.F.EDGE_ATTRS)
NATIVE = counts.F
WIDTHS = (1024, 2048)


def frozen_source(audit, endpoint):
    panel = load_plan(audit)["panels"][endpoint]
    source = panel["source"]
    table, keys, heavy, ok = source_data(audit, source)
    frame, vocab = panel_data(audit, endpoint, panel, table)
    keep = ok[frame.source_position.to_numpy()]
    pos = frame.source_position.to_numpy()[keep]
    assert len(np.unique(pos)) == len(pos)
    return dict(endpoint=endpoint, source=source, table=table, keys=keys[pos, 0],
                ok=ok, pos=pos, texts=table.rxn.to_numpy()[pos],
                ids=table.row_id.to_numpy(str)[pos], heavy=heavy[pos],
                labels=frame.y.to_numpy()[keep], vocab=vocab)


def init_native(audit):
    plan = read(audit.use(CEILING / "ENCODING_PLAN.json"))
    for path, digest in plan["files_sha256"].items():
        audit.use(path, digest)
    repair = BASE / "encoder_coverage_repair_v1"
    adapter = WORKSPACE / "SynRFP/Experiments/51_encoder_coverage_repair/adapter.py"
    checks = read(audit.use(repair / "CODE_READY.json"))["sources_sha256"]
    audit.use(adapter, checks[str(adapter)])
    counts.CONTEXT["adapter"] = counts.load("witness88_dative", adapter)
    repair_plan = read(audit.use(repair / "plan.json", plan["input_plans_sha256"][str(repair / "plan.json")]))
    return {x["source_position"] for x in repair_plan["sources"]["tpl_b"]["repair_rows"]}


def decode(text, repaired=False, provenance=False):
    graph = counts.reaction(text, repaired)
    sides = graph.reactant, graph.product
    if provenance:
        from collections import Counter
        env = [TOKENIZER.tokens_with_provenance(side, 2) for side in sides]
        left, right = [Counter({key: len(v) for key, v in q.items()}) for q in env]
    else:
        left, right = [TOKENIZER.tokens_graph(side, 2) for side in sides]
        env = None
    delta, _, _ = NATIVE.token_algebra(left, right)
    return delta, left, right, sides, env


def sign(delta):
    return counts.signed_digest({k: 1 if n > 0 else -1 for k, n in delta.items()})


def bits_for(data, audit):
    return feature(audit, data["source"], "delta", data["ok"],
                   data["table"].row_id.to_numpy(str))[data["pos"]]
