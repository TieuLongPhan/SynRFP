"""Replay Case B and search exact Tanimoto neighbours in the saved fit rows."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import ctypes
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
OLD = WORKSPACE / "SynRFP-chemistry-cases"
REV75 = "9e2c2de9c38d7ad342bbd86e744d5684b3024e2f"
CODE75 = OLD / "Experiments/75_chemistry_cases"
PUB75 = CODE75 / "publication_metadata/chemistry_cases_v1"
sys.path.insert(0, str(CODE75))

from common import BASE, InputAudit, export, feature, load_plan, panel_data, read, source_data, write
from classification import B, DISPLAY_SEEDS, MODE, comparison_inputs, jaccard

ROOT = BASE / "request80_lost_neighbours_v1"
OUT = HERE / "publication_metadata/lost_neighbours_v1"
RNG_SEED = 80202611
STAGE = WORKSPACE / "SynRFP/Experiments/65_default_optimal/publication_metadata/default_optimal_v1/classification/cells.csv"
ARM = ("d", "u", "du")
STATE = {}


def bound_75(audit, path):
    git_bytes = subprocess.check_output(
        ["git", "-C", str(OLD), "show", f"{REV75}:{path.relative_to(OLD)}"]
    )
    return audit.use(path, sha256(git_bytes).hexdigest())


def kernel(packed, labels, train, query, dp, dup):
    matrix = np.ascontiguousarray(packed).view(np.uint64)
    density = np.bitwise_count(matrix).reshape(len(matrix), 2, 16).sum(axis=2).astype(np.int32)
    out = np.empty((len(query), 3, 7), np.int32)
    method = ctypes.CDLL(str(ROOT / "neighbours.so")).nearest
    ptr = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    method.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS"), ptr, ptr,
                       ptr, ctypes.c_int, ptr, ctypes.c_int, ptr, ptr, ptr]
    method.restype = None
    train, query, labels, dp, dup = (np.ascontiguousarray(a, dtype=np.int32)
                                     for a in (train, query, labels, dp, dup))
    assert len(train) > 0 and not np.intersect1d(train, query).size
    method(matrix, density, labels, train, len(train), query, len(query), dp, dup, out)
    for k in {0, len(query)-1}:
        for c, block in enumerate((packed[:, :128], packed[:, 128:], packed)):
            similarity = jaccard(block, query[k], train)
            best = similarity.max()
            ties = train[similarity == best]
            chosen, num, den, n_tied, vote, _, _ = out[k, c]
            pred = dp[k] if c == 0 else dup[k]
            assert (chosen, num/den, n_tied, vote) == (ties[0], best, len(ties), np.sum(labels[ties] == pred))
    return out


def record(row, row_id, seed, result, true, dp, dup, pos, y, ok, ids, n_fit, pool_size):
    output = dict(r_id=row_id, source_position=int(pos[row]), display_seed=seed,
                  n_fit=n_fit, n_eligible_controls=pool_size)
    for c, (j, num, den, ties, tie_matches, cross_n, cross_d) in enumerate(result):
        arm = ARM[c]
        pred = dp if c == 0 else dup
        output.update({f"{arm}_neighbour_r_id": ids[j], f"{arm}_neighbour_source_position": int(pos[j]),
                       f"{arm}_similarity": num/den, f"{arm}_n_tied": int(ties),
                       f"{arm}_ties_matching_prediction": int(tie_matches),
                       f"{arm}_prediction_matches_neighbour": bool(y[j] == pred),
                       f"{arm}_neighbour_matches_true": bool(y[j] == true),
                       f"{arm}_neighbour_encoded": bool(ok[j])})
        if c < 2:
            output[f"{ARM[1-c]}_similarity_to_{arm}_neighbour"] = cross_n/cross_d
    return output


def endpoint_run(endpoint):
    audit = InputAudit()
    panel = STATE["panels"][endpoint]
    source, _, _, valid = source_data(audit, panel["source"])
    data, vocab = panel_data(audit, endpoint, panel, source)
    pos = data.source_position.to_numpy()
    ids = data.row_id.to_numpy(str)
    y = data.y.to_numpy()
    ok = valid[pos]
    packed = feature(audit, panel["source"], "delta_union", valid, source.row_id.to_numpy(str))[pos]
    bits_d = feature(audit, panel["source"], "delta", valid, source.row_id.to_numpy(str))[pos]
    np.testing.assert_array_equal(bits_d[:, :128] ^ bits_d[:, 128:], packed[:, :128])
    candidate_path = STATE["local"] / f"{endpoint}.json"
    previous = read(audit.use(candidate_path))
    old = {(row["source_position"], row["r_id"]): row for row in previous["candidates"]}
    n_case = STATE["lengths"][endpoint]
    assert len(old) == len(previous["candidates"]) == n_case
    saved, votes, n_test = {}, np.zeros(len(y), int), np.zeros(len(y), int)
    for seed in range(25):
        test, d, du, fit = comparison_inputs(audit, endpoint, seed, panel, data, STATE["comparisons"])
        assert np.all((d >= 0) & (du >= 0)) and not np.intersect1d(fit, test).size
        n_test[test] += 1
        votes[test] += (d == y[test]) & (du != y[test])
        saved[seed] = (test, d, du, fit)
    np.testing.assert_array_equal(n_test, 5)
    eligible = np.flatnonzero(ok & (votes >= 3))
    assert len(eligible) == n_case and {(pos[i], ids[i]) for i in eligible} == set(old)
    reserved = np.zeros(len(y), bool)
    reserved[eligible] = True
    by_seed = defaultdict(list)
    for i in eligible:
        case = old[pos[i], ids[i]]
        assert case["n_D_right_DU_wrong"] == votes[i] and case["n_classes_merged_only_1024"] == 0
        display = None
        for seed in DISPLAY_SEEDS:
            test, d, du, _ = saved[seed]
            j = np.searchsorted(test, i)
            if j < len(test) and test[j] == i and d[j] == y[i] and du[j] != y[i]:
                display = seed
                break
        assert display == case["display_seed"]
        assert (case["true_label"], case["pred_D"], case["pred_DU"]) == (vocab[y[i]], vocab[d[j]], vocab[du[j]])
        by_seed[display].append(i)
    rng = np.random.default_rng(RNG_SEED + B.index(endpoint))
    public, control, extra = [], [], {}
    published = {(int(row["source_position"]), row["r_id"]) for row in STATE["shown"] if row["endpoint"] == endpoint}
    for seed, bad in sorted(by_seed.items()):
        test, d, du, fit = saved[seed]
        good = test[ok[test] & ~reserved[test] & (d == y[test]) & (du == y[test])]
        controls = rng.choice(good, size=len(bad), replace=False)
        reserved[controls] = True
        query = np.r_[bad, controls]
        index = np.searchsorted(test, query)
        np.testing.assert_array_equal(test[index], query)
        dp, dup = d[index], du[index]
        train = fit[np.lexsort((fit, pos[fit]))]
        result = kernel(packed, y, train, query, dp, dup)
        for k, i in enumerate(query):
            row = dict(endpoint=endpoint, **record(i, ids[i], seed, result[k], y[i], dp[k], dup[k],
                                                   pos, y, ok, ids, len(train), len(good)))
            if k < len(bad):
                row["n_D_right_DU_wrong"] = int(votes[i])
                public.append(row)
            else:
                control.append(row)
            key = (int(pos[i]), ids[i])
            if k < len(bad) and key in published:
                j = result[k, 1, 0]
                extra[key] = dict(u_neighbour_rxn=source.rxn.iloc[pos[j]], u_neighbour_class=vocab[y[j]])
    shown = []
    indexed = {(row["source_position"], row["r_id"]): row for row in public}
    for original in STATE["shown"]:
        if original["endpoint"] == endpoint:
            key = (int(original["source_position"]), original["r_id"])
            case = old[key]
            for field in ("display_seed", "rxn", "true_label", "pred_D", "pred_DU"):
                assert str(case[field]) == original[field]
            shown.append(dict(original, **{k: v for k, v in indexed[key].items()
                                           if k not in original}, **extra[key]))
    assert len(shown) == 12 and len(public) == len(control) == n_case
    for path, expected in previous["inputs_sha256"].items():
        if path in audit.pins:
            assert audit.pins[path] == expected
    print(endpoint, len(public), "losses and matching controls", flush=True)
    return public, control, shown, audit.pins


def summarize(endpoint, arm, rows):
    values = pd.DataFrame(rows)
    summary = dict(endpoint=endpoint, population=arm, n_rows=len(values))
    for c in ARM:
        summary[f"{c}_prediction_match_share"] = values[f"{c}_prediction_matches_neighbour"].mean()
        summary[f"{c}_true_match_share"] = values[f"{c}_neighbour_matches_true"].mean()
        summary[f"{c}_similarity_median"] = values[f"{c}_similarity"].median()
        summary[f"{c}_tied_share"] = values[f"{c}_n_tied"].gt(1).mean()
        summary[f"{c}_any_tie_prediction_match_share"] = values[f"{c}_ties_matching_prediction"].gt(0).mean()
        summary[f"{c}_uniform_tie_prediction_match_share"] = (
            values[f"{c}_ties_matching_prediction"]/values[f"{c}_n_tied"]).mean()
    summary["d_similarity_to_u_neighbour_median"] = values.d_similarity_to_u_neighbour.median()
    summary["u_similarity_to_d_neighbour_median"] = values.u_similarity_to_d_neighbour.median()
    summary["u_label_differs_from_true_share"] = (~values.u_neighbour_matches_true).mean()
    wrong_u = values.loc[~values.u_neighbour_matches_true, "u_prediction_matches_neighbour"]
    summary["u_prediction_match_given_wrong_neighbour"] = wrong_u.mean() if len(wrong_u) else None
    return summary


def note(report):
    loss, both = report[-2:]
    share = loss["u_prediction_match_share"]
    conditional = loss["u_prediction_match_given_wrong_neighbour"]
    if share > .5:
        verdict = "Nearest pooled neighbours account for a majority of these selected errors under this diagnostic."
    else:
        verdict = "Nearest pooled neighbours explain a minority of these errors under this diagnostic."
    (OUT / "lost_neighbours_note.md").write_text(
        "# Nearest training neighbours of the Case B losses\n\n"
        f"All **{loss['n_rows']:,} eligible losses** were evaluated on their Request 75 display seeds, "
        f"with **{both['n_rows']:,} distinct both-correct controls** sampled in matching endpoint and split counts. "
        "The five held-out votes, saved predictions, training-set identities and 72 published examples were replayed. "
        "The search used optimizer-training rows and excluded the internal validation rows. No models were fitted.\n\n"
        f"{verdict} D+U's incorrect prediction agrees with the nearest-by-U label in "
        f"**{share:.1%}** of losses and with the nearest-by-D+U label in **{loss['du_prediction_match_share']:.1%}**. "
        f"By comparison, D+U agrees with the nearest-by-U label in **{both['u_prediction_match_share']:.1%}** "
        f"of both-correct controls. In **{loss['u_label_differs_from_true_share']:.1%}** of losses the nearest U "
        f"label differs from the actual class. Within this subset, **{conditional:.1%}** match the wrong prediction. "
        f"Allowing any exact top-score tie raises the loss-set U agreement to "
        f"**{loss['u_any_tie_prediction_match_share']:.1%}**.\n\n"
        f"For losses the median U similarity to the nearest U neighbour was **{loss['u_similarity_median']:.3f}**, "
        f"with median D similarity **{loss['d_similarity_to_u_neighbour_median']:.3f}** to those same neighbours. "
        f"The median D similarity to the nearest D neighbour was **{loss['d_similarity_median']:.3f}**, "
        f"with median U similarity **{loss['u_similarity_to_d_neighbour_median']:.3f}** to those neighbours.\n\n"
        "Controls show the prevalence of correct neighbour agreement when both models succeed. "
        "The diagnostic uses joint two-side pooled context and exact Tanimoto within the archived sketches. "
        "It cannot localize a signal to substrates alone or establish that the neighbour caused the MLP prediction. "
        "The D model used 2,048 bits. Nearest-by-D uses the 1,024-bit D block of D+U, as specified. "
        "Equal scores choose the lowest source position, then the lowest panel position. "
        "Two all-zero sketches have similarity 1. Archived zero placeholders are retained in fit candidate sets and flagged.\n\n"
        "The loss CSV includes neighbour identifiers, match indicators and similarities. "
        "The 72 previously shown cases also include the nearest U reaction and its class for drawing. "
        "Scores for the reference reactions are reported in aggregate.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", default=6, type=int)
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcc", "-O3", "-march=native", "-fPIC", "-shared",
                    str(HERE / "neighbours.c"), "-o", str(ROOT / "neighbours.so")], check=True)
    audit = InputAudit()
    original = read(bound_75(audit, PUB75 / "AUDIT.json"))
    release = read(bound_75(audit, PUB75 / "COMPLETE.json"))
    for name in ("common.py", "classification.py"):
        bound_75(audit, CODE75 / name)
        audit.use(CODE75 / name, original["code_sha256"][name])
    local = BASE / "chemistry_cases_request75_v1"
    receipt = read(audit.use(local / "classification_COMPLETE.json"))
    assert receipt["counts"] == original["source_counts"]["classification_COMPLETE"]["counts"]
    lengths = {r["endpoint"]: r["witness_qualified"] for r in receipt["counts"] if r["case"] == "B"}
    assert set(lengths) == set(B) and sum(lengths.values()) == 2308
    with audit.use(PUB75 / "case_b.csv", release["files_sha256"]["case_b.csv"]).open(newline="") as stream:
        shown = list(csv.DictReader(stream))
    assert len(shown) == 72
    with audit.use(STAGE, original["inputs_sha256"][str(STAGE)]).open(newline="") as stream:
        rows = {(r["dataset"], int(float(r["seed"])), r["method"]): r for r in csv.DictReader(stream)
                if r["model"] == "mlp" and r["method"] in MODE}
    STATE.update(comparisons=rows, shown=shown, lengths=lengths, local=local, panels=load_plan(audit)["panels"])
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(endpoint_run, B))
    losses, controls, drawings, report = [], [], [], []
    for endpoint, (left, right, examples, hashes) in zip(B, results, strict=True):
        losses.extend(left)
        controls.extend(right)
        drawings.extend(examples)
        report.extend((summarize(endpoint, "loss", left), summarize(endpoint, "both_correct", right)))
        for path, expected in hashes.items():
            if path in audit.pins:
                assert audit.pins[path] == expected
            audit.pins[path] = expected
    assert len(losses) == len(controls) == 2308 and len(drawings) == 72
    report.extend((summarize("pooled", "loss", losses), summarize("pooled", "both_correct", controls)))
    export(OUT / "lost_neighbours.csv", losses)
    export(OUT / "lost_neighbours_shown.csv", drawings)
    export(OUT / "lost_neighbours_summary.csv", report)
    note(report)
    files = {path.name: sha256(path.read_bytes()).hexdigest() for path in OUT.iterdir()
             if path.name != "COMPLETE.json"}
    write(OUT / "COMPLETE.json", dict(status="complete", loss_rows=len(losses), control_rows=len(controls),
          shown_rows=len(drawings), summary_rows=len(report), request_75_revision=REV75,
          request_80_revision="7e4a1f0aa91731ef20cc886a6eec7ff127b1f99a", rng_seed=RNG_SEED,
          versions={name: version(name) for name in ("numpy", "pandas", "rdkit")},
          compiler=subprocess.check_output(["gcc", "-dumpfullversion"], text=True).strip(),
          code_sha256={name: sha256((HERE/name).read_bytes()).hexdigest() for name in ("audit.py", "neighbours.c")},
          inputs_sha256=audit.pins, files_sha256=files))
    print("Complete", len(losses), len(controls), len(drawings), flush=True)


if __name__ == "__main__":
    main()
