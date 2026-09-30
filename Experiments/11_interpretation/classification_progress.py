"""Write an explicit, coverage-aware classification progress snapshot."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
sys.path.insert(0, str(EXP))
from synrfp_exp.config import load_config, expand_fingerprints
from importlib.machinery import SourceFileLoader

QUEUE = SourceFileLoader("cpu_queue", str(EXP / "03_classification" / "cpu_benchmark_queue.py")).load_module()


def count_results(root):
    counts = Counter()
    for p in root.rglob("*__*.json"):
        model = p.stem.rsplit("__", 1)[-1]
        if model in {"knn", "mlp", "xgb"}:
            counts[model] += 1
    return dict(counts)


def completion(observed, expected):
    return {model: {
        "completed": int(observed.get(model, 0)),
        "expected": int(total),
        "fraction": round(observed.get(model, 0) / total, 6),
        "percent": round(100 * observed.get(model, 0) / total, 2),
    } for model, total in expected.items()}


def protocol_expected(cfg, model):
    """Count benchmark cells from the protocol when caches are absent.

    The benchmark inventory normally derives availability from ``work/``
    feature and table caches.  Those caches are intentionally local-only and
    may be removed after a completed run, so progress reporting must still be
    able to validate the preserved result tree from the checked-in protocol.
    """
    fps = expand_fingerprints(cfg, cfg["benchmark"]["fingerprints"])
    total = 0
    for dataset in cfg["datasets"]:
        for scheme, seeds in cfg["benchmark"]["schemes"].items():
            extra = expand_fingerprints(
                cfg,
                cfg.get("rxnfp_dataset_fingerprints", {}).get(dataset, []),
            ) if scheme == "fixed" else []
            usable_seeds = [
                seed for seed in seeds
                if not (model == "knn" and scheme == "fixed" and seed != min(seeds))
            ]
            total += len(usable_seeds) * (len(fps) + len(extra))
    return total


def main():
    cfg = load_config(str(EXP / "config.yaml"))
    expected = {}
    missing_by_dataset = {}
    for model in ("knn", "xgb"):
        paths, _ = QUEUE.expected_results(cfg, model)
        missing = QUEUE.missing_results(paths, model)
        expected[model] = len(paths)
        missing_by_dataset[model] = dict(Counter(Path(p).parent.parent.name for p in missing))
        if expected[model] == 0:
            # ``Experiments/work`` is a rebuildable local cache and is absent
            # in a cleaned checkout.  Preserve protocol-level completeness
            # reporting from the result artifacts alone.
            expected[model] = protocol_expected(cfg, model)
    result_root = EXP / "03_classification" / "results"
    benchmark_observed = count_results(result_root / "benchmark")
    factorial_observed = count_results(result_root / "factorial")
    snapshot = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {"expected": expected, "observed": benchmark_observed,
                      "completion": completion(benchmark_observed, expected),
                      "missing_by_dataset": missing_by_dataset},
        "factorial": {"expected_per_model": 8100,
                      "observed": factorial_observed,
                      "completion": completion(factorial_observed,
                                                 {m: 8100 for m in ("knn", "mlp", "xgb")})},
    }
    out = HERE / "results" / "classification_progress.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
