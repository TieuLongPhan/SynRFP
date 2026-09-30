#!/usr/bin/env python
"""E6. Similarity search: nearest training reactions of each test reaction.

Retrieval needs no model and no k for clustering: it asks whether the
fingerprint's native similarity (Tanimoto for bits, cosine otherwise) ranks
same-class reactions first. Label agreement is measured at every hierarchy
level, e.g. whether the top hit of an enzymatic reaction shares its EC1, EC2
and EC3 class.

Outputs: 06_retrieval/results/<dataset>/<fingerprint>.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Experiments/
import argparse

from synrfp_exp import metrics
from synrfp_exp.config import (expand_fingerprints, load_config, log,
                               setup_logging, write_json, results_dir)
from synrfp_exp.data import dataset_table
from synrfp_exp.features import load
from synrfp_exp.models import similarity_topk
from synrfp_exp.scopes import full_scope
from synrfp_exp.splits import TEST, make_split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--fingerprints", nargs="*")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    rc = cfg["retrieval"]
    kmax = max(rc["k"])
    for dataset in args.datasets or rc["datasets"]:
        spec = cfg["datasets"][dataset]
        try:
            ds = dataset_table(cfg, dataset)
        except FileNotFoundError as e:
            log.error("%s skipped: %s", dataset, e)
            continue
        scheme = rc.get("scheme", "fixed")
        try:
            code = make_split(cfg, dataset, scheme, 0)
        except (FileNotFoundError, ValueError) as e:
            log.error("%s skipped: %s", dataset, e)
            continue
        q, db = code == TEST, code != TEST
        sc = full_scope(cfg, dataset)
        levels = spec.get("hierarchy") or [spec["label"]]
        for fp in expand_fingerprints(cfg, args.fingerprints or rc["fingerprints"]):
            out = results_dir("06_retrieval") / dataset / f"{fp}.json"
            if out.exists():
                continue
            try:
                feats = load(cfg, sc.name, fp)
            except FileNotFoundError as e:
                log.warning("skip %s: %s", fp, e)
                continue
            rows = sc.feat_rows
            idx, _ = similarity_topk(feats.dense(rows[q]), feats.dense(rows[db]), kmax,
                                     tanimoto=feats.is_bits)
            rec = {"dataset": dataset, "fingerprint": fp, "scheme": scheme,
                   "n_query": int(q.sum()), "n_database": int(db.sum())}
            for lev in levels:
                col = "label" if lev == spec["label"] else lev
                y = ds[col].to_numpy()
                yq, yn = y[q], y[db][idx]
                for k in rc["k"]:
                    for name, v in metrics.retrieval(yq, yn, k).items():
                        rec[f"{lev}__{name}"] = v
            write_json(out, rec)
            log.info("%s %s p@1=%.3f", dataset, fp, rec[f"{spec['label']}__p@1"])


if __name__ == "__main__":
    main()
