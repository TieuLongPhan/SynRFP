#!/usr/bin/env python
"""E2. Compute and cache fingerprints.

--study benchmark   SynRFP default + baselines on every full dataset
--study factorial   the 162-point SynRFP grid (large datasets subsampled)
--study analysis    fingerprints used by clustering / retrieval / robustness
--study all         everything above

RXNFP features are produced in a separate environment (external/).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Experiments/
import argparse

from synrfp_exp.config import (expand_fingerprints, load_config, log,
                               setup_logging)
from synrfp_exp.features import EXTERNAL, compute
from synrfp_exp.scopes import factorial_scope, full_scope


def featurize(cfg, datasets, fps, scope_fn):
    done = set()
    for name in datasets:
        try:
            sc = scope_fn(cfg, name)
        except FileNotFoundError as e:
            log.error("%s skipped: %s", name, e)
            continue
        if sc.name in done:
            continue
        compute(cfg, sc.name, sc.rsmis, [f for f in fps if f not in EXTERNAL])
        done.add(sc.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--study", default="all",
                    choices=["benchmark", "factorial", "analysis", "all"])
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--fingerprints", nargs="*")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)

    if args.study in ("benchmark", "all"):
        fps = expand_fingerprints(cfg, args.fingerprints or cfg["benchmark"]["fingerprints"])
        featurize(cfg, args.datasets or list(cfg["datasets"]), fps, full_scope)
    if args.study in ("factorial", "all"):
        fps = expand_fingerprints(cfg, args.fingerprints or ["synrfp_grid"])
        featurize(cfg, args.datasets or cfg["factorial"]["datasets"], fps, factorial_scope)
    if args.study in ("analysis", "all"):
        names = list(dict.fromkeys(cfg["clustering"]["datasets"] + cfg["retrieval"]["datasets"]))
        fps = expand_fingerprints(
            cfg, args.fingerprints or (cfg["clustering"]["fingerprints"]
                                       + cfg["retrieval"]["fingerprints"]))
        featurize(cfg, args.datasets or names, fps, full_scope)


if __name__ == "__main__":
    main()
