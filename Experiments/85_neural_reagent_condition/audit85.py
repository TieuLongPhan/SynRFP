"""Recompute native-split specialist metrics using saved predictions and pinned inputs."""
import ast
from collections import Counter
import csv
from functools import lru_cache
import gzip
import hashlib
import importlib.util
from importlib.metadata import version
import io
import json
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parents[2] / "SynRFP"
RUNS = Path.home() / ".local/share/synrfp-runs"
SPECIALISTS = RUNS / "specialist_references_v1"
OUTPUT = HERE / "publication_metadata/native_audit_v1"
VENDOR = SPECIALISTS / "vendor/Parrot"
NATIVE_PYTHON = Path.home() / "anaconda3/envs/rxnfp/bin/python"
SLOTS = ("catalyst1", "solvent1", "solvent2", "reagent1", "reagent2")
ROLES = ("catalyst", "solvent", "solvent", "reagent", "reagent")
KEYS = ("exact_set_accuracy", "micro_f1", "sample_jaccard")
STUDIES = {"t5chem": "reagent", "parrot": "condition"}
NATIVE_SEEDS = {"t5chem": (42, 43, 44), "parrot": (42, 43)}
EXPECTED_ROWS = {"t5chem": 14215, "parrot": 65946}
PINS = {}


def read(path):
    return json.loads(Path(path).read_text())


def pin(path, expected=None):
    path = Path(path)
    key = str(path)
    if key not in PINS:
        with path.open("rb") as source:
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        PINS[key] = digest.hexdigest()
    if expected is not None and expected != PINS[key]:
        raise ValueError(f"Changed frozen dependency {path}")
    return PINS[key]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def ordered_hash(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_frame(table, scope):
    columns = ["row_id", "source_position", "native_split", "eligible"]
    if scope == "condition":
        columns.append("raw_source")
    frame = pd.read_csv(table, usecols=columns, dtype={"row_id": str}, keep_default_na=False)
    np.testing.assert_array_equal(frame.source_position, np.arange(len(frame)))
    assert frame.row_id.is_unique
    return frame


def compare_ids(comparison, scored, reference, **extra):
    a, b = set(scored), set(reference)
    assert len(a) == len(scored) and len(b) == len(reference)
    return dict(comparison=comparison, scored_rows=len(a), reference_rows=len(b),
                intersection_rows=len(a & b), scored_only=len(a - b), reference_only=len(b - a),
                scored_sorted_id_sha256=ordered_hash(sorted(a)), reference_sorted_id_sha256=ordered_hash(sorted(b)),
                **extra)


def inputs(kind):
    study = STUDIES[kind]
    source = RUNS / f"{study}_prediction_v1"
    specialist = SPECIALISTS / kind
    parent = source / "plan.json"
    plan = specialist / "plan.json"
    plan_digest = pin(plan, read(specialist / "plan_seal.json")["sha256"])
    p, f = read(plan), read(parent)
    pin(parent, read(source / "plan_seal.json")["sha256"])
    assert pin(parent) == p["parent_plan_sha256"] and p["kind"] == kind
    table = source / "inputs/reactions.csv.gz"
    targets = source / "inputs/targets.json"
    files = (table, targets)
    for path in files:
        pin(path, p["pinned_files"][str(path)])
        pin(path, f["pinned_files"][str(path)])
    model_dirs = {}
    selected = None
    for seed in NATIVE_SEEDS[kind]:
        name = f"native_seed{seed}"
        local, = (x for x in p["splits"] if x["name"] == name)
        parent_split, = (x for x in f["splits"] if x["name"] == name)
        assert local == parent_split
        sel = local["path"]
        pin(sel, p["pinned_files"][sel])
        pin(sel, f["pinned_files"][sel])
        with np.load(sel, allow_pickle=False) as data:
            split = {key: data[key] for key in ("train", "validation", "test")}
        expected = EXPECTED_ROWS[kind]
        assert len(split["test"]) == local["test_rows"] == expected
        assert len(np.unique(split["test"])) == expected and not np.intersect1d(split["test"], split["train"]).size
        if selected is not None:
            for key in split:
                np.testing.assert_array_equal(selected[key], split[key])
        selected = split
        model_dirs[seed] = specialist / "models" / f"native_seed{seed}"
    frame = load_frame(table, study)
    assert len(frame) == p["rows"]
    eligible = frame.iloc[selected["test"]]
    assert eligible.eligible.all() and eligible.native_split.eq("test").all()
    np.testing.assert_array_equal(selected["test"], np.flatnonzero(frame.eligible & frame.native_split.eq("test")))
    references = read(targets)
    assert len(references) == len(frame)
    return dict(source=source, specialist=specialist, model_dirs=model_dirs, frame=frame, split=selected,
                references=[references[int(i)] for i in selected["test"]], plan=p, plan_sha256=plan_digest,
                ids=eligible.row_id.tolist(), parent_refs=f)


def predictions(model_dir, positions, seed):
    path = model_dir / "predictions.json.gz"
    rec = model_dir / "COMPLETE.json"
    pin(rec)
    receipt = read(rec)
    pin(path, receipt["artifacts_sha256"][str(path)])
    assert receipt["phase"] == "complete" and receipt["seed"] == seed and receipt["protocol"] == "native"
    assert receipt["test_rows"] == len(positions)
    with gzip.open(path, "rt") as stream:
        pred = json.load(stream)
    np.testing.assert_array_equal(positions, pred["rows"])
    assert len(pred["predictions"]) == len(positions)
    return pred["rows"], pred["predictions"], receipt


class DataPickle(pickle.Unpickler):
    """Only read the publisher's primitive label lists and index dictionaries."""

    def find_class(self, module_name, class_name):
        raise ValueError(f"Object instantiation prohibited in release labels: {module_name}.{class_name}")


def official_test(data):
    parent = data["source"]
    raw = parent / "raw/USPTO_condition.csv"
    prov = parent / "source_provenance.json"
    expected_pins = data["parent_refs"]["pinned_files"]
    pin(prov, expected_pins[str(prov)])
    provenance = read(prov)
    pin(raw, provenance["raw_csv_sha256"])
    pin(raw, data["plan"]["pinned_files"][str(raw)])
    file, = provenance["downloaded_files"]
    archive = file["path"]
    pin(archive, file["sha256"])
    frame = data["frame"]
    with raw.open(newline="") as stream:
        reader = csv.DictReader(stream)
        entries = {}
        rows = set(data["split"]["test"])
        for position, (meta, item) in enumerate(zip(frame.itertuples(index=False), reader, strict=True)):
            assert meta.raw_source == item["source"] and meta.native_split == item["dataset"]
            if item["dataset"] == "test":
                assert meta.row_id == f"uspto-condition-{position}"
                entries[position] = tuple(item[c] for c in reader.fieldnames)
    assert len(entries) == provenance["inventory"]["native_raw_counts"]["test"] == 68075
    assert rows <= entries.keys()
    ids = [f"uspto-condition-{i}" for i in entries]
    offset = [reader.fieldnames.index(column) for column in SLOTS]
    with zipfile.ZipFile(archive) as z:
        prefix = "USPTO_condition_final/USPTO_condition"
        with z.open(prefix + ".csv") as stream:
            assert hashlib.sha256(stream.read()).hexdigest() == pin(raw)
        with z.open(prefix + "_testset.csv") as stream:
            test = csv.reader(io.TextIOWrapper(stream, encoding="utf-8", newline=""))
            assert next(test) == reader.fieldnames
            assert Counter(map(tuple, test)) == Counter(entries.values())
        with z.open(prefix + "_alldata_idx.pkl") as stream:
            _, label_to_id = DataPickle(stream).load()
        with z.open(prefix + "_condition_labels.pkl") as stream:
            encoded = DataPickle(stream).load()
    assert len(encoded) == len(frame)
    references = {i: tuple(row[j] for j in offset) for i, row in entries.items()}
    for i, slots in references.items():
        assert encoded[i][0] == label_to_id["[BOS]"] and encoded[i][-1] == label_to_id["[EOS]"]
        assert list(encoded[i][1:-1]) == [label_to_id[s] for s in slots]
    audit = parent / "input_audit.json"
    pin(audit, expected_pins[str(audit)])
    audit = read(audit)
    missing = entries.keys() - rows
    parse = {x["position"] for x in audit["refusal_records"]}
    overlap = {x["position"] for x in audit["overlap_records"]}
    assert missing == ((parse | overlap) & entries.keys())
    assert not parse & overlap
    counts = dict(reference_only_parse_refusal=len(missing & parse),
                  reference_only_condition_also_in_input=len(missing & overlap))
    return [references[i] for i in data["split"]["test"]], ids, label_to_id, counts


def official_helper():
    """Replay author arithmetic in the specialist PyTorch environment."""
    import torch

    model = VENDOR / "models/parrot_model.py"
    tree = ast.parse(model.read_text())
    model_cls, = (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ParrotConditionPredictionModel")
    method, = (n for n in model_cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_accuracy_for_one")
    ns = {"pd": pd}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(model), "exec"), ns)
    scorer = ns[method.name]
    hits = np.zeros(6, dtype=int)
    for pattern, count in json.load(sys.stdin):
        pred = torch.tensor([pattern], dtype=torch.long)
        ref = torch.ones(5, dtype=torch.long)
        data = scorer(None, pred, ref, topk_get=[1])
        np.testing.assert_array_equal(data[1], [*pattern, all(pattern)])
        hits += data[1].to_numpy(dtype=int) * count
    print(json.dumps(dict(hits=hits.tolist(), torch=torch.__version__, pandas=pd.__version__, python=sys.version)))


def official_score(raw, references, label_to_id):
    """Use the pinned author helper on every distinct observed slot equality pattern."""
    for path in ("models/parrot_model.py", "models/utils.py", "test_parrot_model.py",
                 "configs/config_uspto_condition.yaml"):
        pin(VENDOR / path)
    exact_patterns = Counter()
    for a, b in zip(raw, references, strict=True):
        assert len(a) == len(b) == 5
        exact_patterns[tuple(label_to_id.get(x, -1) == label_to_id[y] for x, y in zip(a, b, strict=True))] += 1
    process = subprocess.run([NATIVE_PYTHON, Path(__file__), "--author-helper"],
                             input=json.dumps(list(exact_patterns.items())), text=True, check=True,
                             stdout=subprocess.PIPE)
    result = json.loads(process.stdout)
    hits = result.pop("hits")
    n = len(raw)
    return dict(parrot_top1_overall=hits[-1] / n, **{f"top1_{s}": v / n for s, v in zip(SLOTS, hits[:5], strict=True)}), result


def main():
    RDLogger.DisableLog("rdApp.*")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    driver_path = ORIGINAL / "Experiments/35_specialist_references/t5chem/run.py"
    driver = module("specialist_replay_85", driver_path)
    parser_path = ORIGINAL / "Experiments/23_reagent_prediction/study.py"
    parser = module("parent_reagent_85", parser_path)
    normalizer_path = ORIGINAL / "Experiments/35_specialist_references/parrot/normalize.py"
    normalize = module("native_normalize_85", normalizer_path)
    canonical_fragment = lru_cache(maxsize=8192)(parser.canonical_fragment)

    @lru_cache(maxsize=8192)
    def canonical_slot(value):
        return ".".join(sorted(canonical_fragment(p) for p in value.split("."))) if value else ""

    @lru_cache(maxsize=8192)
    def role_components(slots):
        return sorted({json.dumps(["condition-role-component-v1", role, part], separators=(",", ":"), ensure_ascii=False)
                       for role, slot in zip(ROLES, slots, strict=True) if slot for part in canonical_slot(slot).split(".")})

    all_overlap = []
    t5, parrot, diagnostics = [], [], []
    for kind in STUDIES:
        data = inputs(kind)
        plan, test = data["plan"], data["split"]["test"]
        for path in (driver_path, parser_path, normalizer_path):
            pin(path, plan["pinned_files"][str(path)])
        ref = data["references"]
        sha = ordered_hash(sorted(data["ids"]))
        if kind == "parrot":
            commit = subprocess.check_output(["git", "-C", VENDOR, "rev-parse", "HEAD"], text=True).strip()
            assert commit == plan["official_commit"]
            raw_ref, author_ids, author_label_ids, reasons = official_test(data)
            slots_path = data["specialist"] / "slots.json"
            pin(slots_path, plan["pinned_files"][str(slots_path)])
            slots = read(slots_path)
            canonical_ref = [tuple(canonical_slot(value) for value in row) for row in raw_ref]
            assert canonical_ref == [tuple(slots[i]) for i in test]
            assert [role_components(row) for row in canonical_ref] == ref

        for seed, model_dir in data["model_dirs"].items():
            positions, pred, receipt = predictions(model_dir, test, seed)
            scored_ids = data["frame"].iloc[positions].row_id.tolist()
            stored = [p["components"][0] for p in pred]
            reported = driver.set_scores(stored, ref)
            for key, value in reported.items():
                assert np.isclose(value, receipt["metrics"][key], atol=1e-12, rtol=0), key
            base = dict(seed=seed, n_test_rows=len(test), test_sorted_id_sha256=sha)
            all_overlap.append(compare_ids(f"{kind}_vs_fingerprint_native", scored_ids, data["ids"], seed=seed))
            if kind == "t5chem":
                top = [{"raw": [p["raw"][0]]} for p in pred]
                recomputed = [p["components"][0] for p in normalize.normalize("reagent_predictions", top)]
                assert recomputed == stored
                t5.append(dict(row_type="seed", **base, **{key: reported[key] for key in KEYS}))
                continue

            all_overlap.append(compare_ids("parrot_vs_published_author_test", scored_ids, author_ids, seed=seed, **reasons))
            raw = [tuple(p["raw"][0]) for p in pred]
            assert all(len(row) == 5 and all(isinstance(slot, str) for slot in row) for row in raw)
            rescored = [role_components(row) for row in raw]
            assert rescored == stored
            canonical_pred = [tuple(canonical_slot(value) for value in row) for row in raw]
            canonical_ordered = sum(a == b for a, b in zip(canonical_pred, canonical_ref, strict=True)) / len(test)
            strict, author_env = official_score(raw, raw_ref, author_label_ids)
            variants = [
                dict(scoring_variant="saved_native_role_component_set", **reported),
                dict(scoring_variant="rescore_canonical_role_component_set_drop_empty", **driver.set_scores(rescored, ref)),
                dict(scoring_variant="author_ordered_five_slot_top1", **strict),
                dict(scoring_variant="canonical_ordered_five_slot_top1", parrot_top1_overall=canonical_ordered),
            ]
            parrot.extend(dict(**base, **variant) for variant in variants)
            mode, mode_count = Counter(raw).most_common(1)[0]
            history_path = model_dir / "history.json"
            pin(history_path, receipt["artifacts_sha256"][str(history_path)])
            history = read(history_path)
            vocabulary_path = model_dir / "vocabulary.json"
            pin(vocabulary_path, receipt["artifacts_sha256"][str(vocabulary_path)])
            vocabulary = set(read(vocabulary_path))
            unknown = Counter(v for row in canonical_ref for v in row if v not in vocabulary)
            diagnostics.append(dict(seed=seed, n_test_rows=len(test), distinct_top1_tuples=len(set(raw)), modal_top1_rows=mode_count,
                modal_predicted_catalyst=mode[0], modal_predicted_first_solvent=mode[1], modal_predicted_second_solvent=mode[2],
                modal_predicted_first_reagent=mode[3], modal_predicted_second_reagent=mode[4],
                test_rows_with_unseen_canonical_slot=sum(any(v not in vocabulary for v in row) for row in canonical_ref),
                distinct_unseen_canonical_slots=len(unknown), canonical_slot_vocabulary=len(vocabulary)-4,
                test_rows_with_exact_reference_among_saved_five=sum(any(set(x) == set(y) for x in p["components"])
                    for p, y in zip(pred, ref, strict=True)), training_epochs=receipt["epochs"],
                best_epoch=min(history, key=lambda h: h["validation_loss"])["epoch"],
                best_validation_loss=receipt["best_validation_loss"], last_validation_loss=history[-1]["validation_loss"],
                distinct_reference_slot_tuples=len(set(canonical_ref)), reference_components_mean=np.mean(list(map(len, ref)))))

    t5_data = pd.DataFrame(t5)
    for name, reduction in (("mean", statistics.mean), ("sample_sd", statistics.stdev)):
        t5.append(dict(row_type=name, seed="", n_test_rows=t5[0]["n_test_rows"], test_sorted_id_sha256=t5[0]["test_sorted_id_sha256"],
                       **{key: reduction(t5_data[key]) for key in KEYS}))
    table = pd.DataFrame(parrot)
    hits = table.loc[table.scoring_variant.eq("rescore_canonical_role_component_set_drop_empty"), "exact_set_accuracy"]
    top1 = table.loc[table.scoring_variant.eq("author_ordered_five_slot_top1"), "parrot_top1_overall"]
    passed = bool((hits >= .15).all() or (top1 >= .15).all())
    if passed:
        raise ValueError("Saved scores meet the seed-44 threshold. Review admission before publication.")
    for name, rows in (("t5chem_reagent_seeds", t5), ("parrot_condition_audit", parrot),
                       ("test_row_overlap", all_overlap), ("parrot_diagnostics", diagnostics)):
        pd.DataFrame(rows).to_csv(OUTPUT / f"{name}.csv", index=False)
    t = t5[-2]
    cond = all_overlap[-1]
    d = diagnostics[-1]
    note = f"""# Native reagent and condition reference audit

T5Chem is usable as a matched native-split neural reagent reference. Seeds 42–44 score the same
{t['n_test_rows']:,} frozen test rows as the fingerprints, with mean exact-set accuracy
{t['exact_set_accuracy']:.2%}, micro-F1 {t['micro_f1']:.4f} and mean per-row Jaccard
{t['sample_jaccard']:.4f}. The CSV gives sample SD across three optimizer seeds on this one fixed
split. Parrot seeds 42 and 43 score all {cond['scored_rows']:,} fingerprint test rows, a strict subset
of the {cond['reference_rows']:,} author test rows. The {cond['reference_only']:,} omitted author rows
are accounted for by the fingerprint benchmark's frozen eligibility rules
({cond['reference_only_condition_also_in_input']:,} input/condition overlaps and
{cond['reference_only_parse_refusal']:,} component parse refusals). Canonicalizing molecules,
dropping empty slots, and ignoring their order within each role leaves both seeds at
{hits.iloc[0]:.2%} exact set. The author's strict ordered five-slot matcher gives
{top1.iloc[0]:.2%} for both on these same rows. Both saved fits return dichloromethane as the only
top-ranked condition on every reaction. The train-only canonical vocabulary covers every reference
slot on these test rows. This points to collapsed top-ranked predictions in the saved adaptation.
Seed 44 remains at its existing partial checkpoint. Exclude these fits as a published-Parrot
reference. The published Parrot-LM-E obtained 26.91% ordered top-1 on the author test set with an
additional fivefold SMILES augmentation stage [Wang et al., 2023, Table 1](https://yuquanli.github.io/pdf/paper_2023b.pdf).
This audit rescored the local unaugmented adaptation's saved candidates.

The study23 and study29 set metrics preserve the benchmark role-labelled target identities for
conditions. Within-role slot order and repeated components are discarded and an empty union has
Jaccard zero. The author score compares encoded identities in the original five slot positions,
including its empty category. The original helper was replayed from the pinned author source on
every distinct observed five-slot agreement pattern and weighted by its row count. The
canonical-ordered companion measures the effect of canonical SMILES while keeping slot positions.
"""
    (OUTPUT / "neural_reagent_condition_note.md").write_text(note)
    manifest = dict(status="complete", request_revision="e4e6055b897021de9a4ae41a530e513c0cba8e78",
                    output_sha256={p.name: pin(p) for p in sorted(OUTPUT.iterdir()) if p.name != "COMPLETE.json"},
                    input_sha256={k: v for k, v in sorted(PINS.items()) if not k.startswith(str(OUTPUT))},
                    audit_code_sha256=pin(Path(__file__)), packages={p: version(p) for p in ("numpy","pandas","rdkit")},
                    python=sys.version, author_scorer_environment=author_env, native_seed_counts=dict(t5chem=3, parrot=2),
                    seed44_gate=dict(threshold=.15, passed=passed, training_launched=False),
                    parrot_publisher_model="Parrot-LM-E 26.91% ordered top-1, Table 1",
                    literature="https://doi.org/10.34133/research.0231", no_new_fits=True)
    (OUTPUT / "COMPLETE.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"T5Chem {t['exact_set_accuracy']:.4f}, Parrot exact {hits.tolist()}, author top1 {top1.tolist()}, gate {passed}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--author-helper"]:
        official_helper()
    else:
        main()
