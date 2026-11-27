"""Select and certify the requested chemistry witnesses without fitting models."""
from collections import Counter, defaultdict
import gzip
from functools import lru_cache
from importlib.metadata import version
from itertools import combinations
import json
import re

import numpy as np
import pandas as pd
from rdkit import Chem

from common88 import (
    BASE, BRANCH75, HERE, NATIVE, OUTPUT, ROOT, WIDTHS, WORKSPACE, InputAudit,
    bits_for, chemistry, counts, decode, frozen_source, init_native, read,
    rooted_examples, sign, write,
)
from common import Buckets, export, groups, load_plan, packed_ids, panel_data
from synrfp.sketchers.parity_fold import ParityFold

FIELDS = ("endpoint", "pair_id", "side", "r_id", "rxn", "label", "n_heavy_atoms", "scaffold",
          "source_position", "mapped_rxn", "mapping_origin")
STANDARD = "synrfp-wl-r2-b2048-parity-delta"
REVERSE_PANELS = ("schneider_ub", "tpl_b", "ecreact_1", "mech_2")
B_PANELS = ("mech_1", "mech_2", "syntemp_r1", "syntemp_r2", "uspto50k_b", "uspto50k_ub")
SEP = re.compile(r"(?<!-)>")


def jtext(value):
    return json.dumps(value, separators=(",", ":"))


def load_ec_labels(audit):
    root = WORKSPACE / "SynRFP-ecreact-label-map/Experiments/78_clustering_barrier_ceiling"
    path = root / "publication_metadata/clustering_barrier_ceiling_v1/ecreact_labels/ecreact_label_map.csv"
    table = pd.read_csv(audit.use(path), keep_default_na=False, dtype={"ec": str})
    return {str(i): ec for i, ec in zip(table.loc[table.level.eq(1), "index"],
                                       table.loc[table.level.eq(1), "ec"], strict=True)}


def decode_label(data, i, ec):
    raw = data["vocab"][data["labels"][i]]
    return ec[raw] if data["endpoint"] == "ecreact_1" else raw


def row_values(endpoint, pair, side, identifier, reaction, label, position):
    size, scaffold = chemistry(reaction)
    return dict(endpoint=endpoint, pair_id=pair, side=side, r_id=str(identifier), rxn=reaction,
                label=str(label), n_heavy_atoms=size, scaffold=scaffold, source_position=int(position),
                mapped_rxn="", mapping_origin="")


def from_source(data, i, pair, side, ec):
    return row_values(data["endpoint"], pair, side, data["ids"][i], data["texts"][i],
                      decode_label(data, i, ec), data["pos"][i])


def exact_delta(data, i, repaired, provenance=False):
    text = data["texts"][i]
    result = decode(text, data["source"] == "tpl_b" and data["pos"][i] in repaired, provenance)
    assert counts.signed_digest(result[0]) == bytes(data["keys"][i])
    return result


def signs(data, audit):
    base = ROOT / f"{data['endpoint']}_sign"
    rec = read(audit.use(base.with_suffix(".json")))
    for name, digest in rec["code_sha256"].items():
        audit.use(HERE/name, digest)
    with np.load(audit.use(base.with_suffix(".npz"), rec["sha256"]), allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_positions"], data["pos"])
        return saved["keys"]


def magnitude_pairs(audit, used, repaired, ec, source):
    candidates, eligible = [], {}
    for endpoint in ("ecreact_1", "tpl_b"):
        data = source(endpoint)
        signed = signs(data, audit)
        label = data["labels"]
        exact = groups(data["keys"])
        support = groups(signed)
        joint_counts = lambda value: np.bincount(np.unique(np.column_stack((support, value)), axis=0)[:, 0])
        groups_in_scope = np.flatnonzero((joint_counts(exact) > 1) & (joint_counts(label) > 1))
        buckets = Buckets(support)
        n_pairs = 0
        forced_before = forced_after = 0
        for g in groups_in_scope:
            members = buckets.members(g)
            before = defaultdict(Counter)
            for i in members:
                before[exact[i]][label[i]] += 1
            forced_before += len(members) - sum(max(c.values()) for c in before.values())
            forced_after += len(members) - max(Counter(label[members]).values())
            reps = {}
            for i in members:
                size, scaffold = chemistry(data["texts"][i])
                key = (int(exact[i]), int(label[i]), scaffold)
                rank = (size, str(data["ids"][i]), int(data["pos"][i]))
                if key not in reps or rank < reps[key][0]:
                    reps[key] = (rank, i)
            for a, b in combinations(sorted(reps.values()), 2):
                a_rank, i = a
                b_rank, j = b
                if exact[i] == exact[j] or label[i] == label[j]:
                    continue
                a_scaffold, b_scaffold = chemistry(data["texts"][i])[1], chemistry(data["texts"][j])[1]
                if a_scaffold == b_scaffold:
                    continue
                candidates.append(((a_rank[0] + b_rank[0], a_rank, b_rank, endpoint), endpoint, i, j))
            # Inclusion–exclusion counts all underlying row pairs before scaffold filtering.
            n2 = lambda n: n * (n - 1) // 2
            n_pairs += (n2(len(members)) - sum(n2(int(n)) for n in Counter(exact[members]).values())
                        - sum(n2(int(n)) for n in Counter(label[members]).values())
                        + sum(n2(n) for c in before.values() for n in c.values()))
        eligible[endpoint] = dict(encoded_rows=len(label), sign_classes=len(buckets.offset)-1,
                                  mixed_label_multi_count_classes=len(groups_in_scope),
                                  eligible_original_row_pairs=n_pairs,
                                  added_forced_errors=int(forced_after-forced_before))
    assert eligible["ecreact_1"]["added_forced_errors"] == 13
    rows, accepted = [], Counter()
    for _, endpoint, i, j in sorted(candidates):
        if accepted[endpoint] == 4:
            continue
        data = source(endpoint)
        sc = {chemistry(data["texts"][k])[1] for k in (i, j)}
        if sc & used:
            continue
        left = exact_delta(data, i, repaired, provenance=True)
        right = exact_delta(data, j, repaired, provenance=True)
        a, b = left[0], right[0]
        assert a != b and sign(a) == sign(b)
        difference = sorted(token for token in a if a[token] != b[token])
        assert difference and all(a[k] * b[k] > 0 for k in difference)
        accepted[endpoint] += 1
        pair = f"A_{endpoint}_{accepted[endpoint]:02d}"
        for side, (k, own, other) in enumerate(((i, left, right), (j, right, left)), 1):
            row = from_source(data, k, pair, side, ec)
            row.update(differing_tokens=jtext([[f"{t:016x}", own[0][t], other[0][t]] for t in difference]),
                       differing_wl_environments=jtext(rooted_examples(difference, *own[3:])),
                       signed_support_equal=True)
            rows.append(row)
        used.update(sc)
    for endpoint in eligible:
        eligible[endpoint]["exported_pairs"] = accepted[endpoint]
    return rows, eligible


def canonical_sides(text, stereo=True, isotopes=True):
    if len(SEP.split(text)) != 3:
        raise ValueError("Expected a three-field reaction")
    reactants, agents, products = SEP.split(text)
    def compose(*sides):
        components = []
        for side in sides:
            for item in filter(None, side.split(".")):
                mol = Chem.MolFromSmiles(item)
                if mol is None:
                    raise ValueError("Source component refused canonicalization")
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                    if not isotopes:
                        atom.SetIsotope(0)
                if not stereo:
                    Chem.RemoveStereochemistry(mol)
                components.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        return sorted(components)
    return compose(reactants, agents), compose(products, agents)


def reversal_cause(text, delta):
    if delta:
        return "Nonempty signed delta with identical folded vectors in both directions"
    a, b = canonical_sides(text)
    if a == b:
        return "Identical canonical reactant and product component multisets"
    a, b = canonical_sides(text, stereo=False)
    if a == b:
        return "Stereo-only difference under canonical component comparison"
    a, b = canonical_sides(text, stereo=False, isotopes=False)
    if a == b:
        return "Isotope or combined stereo/isotope change outside the benchmark WL attributes"
    return "Distinct canonical sides with equal WL radius-2 token multisets"


def reverse_equal(audit, used, repaired, ec, source, withheld):
    base = BASE / "request86_invariance_v1"
    preparation = read(audit.use(base/"manifest.json"))
    proof = read(audit.use(base/"classical_COMPLETE.json"))
    data = pd.read_csv(audit.use(base/"forms.csv", preparation["forms_sha256"]), keep_default_na=False)
    candidates = []
    for endpoint in REVERSE_PANELS:
        fp = base/f"classical_{endpoint}.npz"
        with np.load(audit.use(fp, proof["outputs_sha256"][fp.name]), allow_pickle=False) as saved:
            x, ok = saved[STANDARD], saved[STANDARD+"__ok"]
        same = np.flatnonzero(ok[0] & ok[2] & np.all(x[0] == x[2], axis=1))
        if not len(same):
            continue
        d = source(endpoint)
        indices = {int(pos): i for i, pos in enumerate(d["pos"])}
        panel = data[data.endpoint.eq(endpoint)].set_index("sample")
        for sample in same:
            r = panel.loc[sample]
            i = indices[int(r.source_position)]
            assert r.original == d["texts"][i] and r.row_id == d["ids"][i]
            delta, a, b, *_ = exact_delta(d, i, repaired)
            reverse = decode(r.reversed)[0]
            assert {k: -v for k, v in delta.items()} == reverse
            for left, right, saved_bits in ((a, b, x[0, sample]), (b, a, x[2, sample])):
                np.testing.assert_array_equal(NATIVE._packed(left, right, "parity", 2048, "delta"), saved_bits)
            item = from_source(d, i, "", 1, ec)
            item.update(sample=int(sample), n_signed_tokens=len(delta),
                        signed_delta=jtext({f"{k:016x}":v for k,v in sorted(delta.items())}),
                        cause=reversal_cause(r.original, delta))
            candidates.append(item)
    rows = []
    for number, row in enumerate(sorted(candidates, key=lambda r: (r["n_heavy_atoms"], r["r_id"], r["source_position"])), 1):
        row["pair_id"] = f"C_{number:02d}"
        if row["scaffold"] in used:
            withheld.append({k:v for k,v in row.items() if k not in ("rxn","mapped_rxn","mapping_origin")} |
                            {"part":"C", "reason":"scaffold already used by the smaller qualifying row"})
        else:
            rows.append(row)
            used.add(row["scaffold"])
    return rows, dict(eligible_reactions=len(candidates), displayed_reactions=len(rows),
                      causes=dict(Counter(r["cause"] for r in candidates)))


def bit_witness(a, b, width):
    fold = ParityFold(bits=width, seed=1)
    def to_bins(delta):
        hits = defaultdict(list)
        for token, value in sorted(delta.items()):
            tag = 2*int(token) + (value < 0)
            hits[fold.bit_index(tag)].append((f"{token:016x}", "+" if value > 0 else "-"))
        return hits
    left, right = to_bins(a), to_bins(b)
    assert {k for k in left if len(left[k]) % 2} == {k for k in right if len(right[k]) % 2}
    return [{"bit_index": k, "left_signed_keys": left[k], "right_signed_keys": right[k],
             "shared_bit_value": len(left[k]) % 2}
            for k in sorted(left.keys() | right.keys()) if left[k] != right[k]]


def fold_cases(audit, used, repaired, ec, source, withheld):
    folder = BRANCH75/"publication_metadata/chemistry_cases_v1"
    seal = read(audit.use(folder/"COMPLETE.json"))
    name = "fold_collision_summary.csv"
    archive = pd.read_csv(audit.use(folder/name, seal["files_sha256"][name]))
    rows, inventory = [], []
    for endpoint in (*B_PANELS, "ecreact_1"):
        data = source(endpoint)
        exact = groups(data["keys"])
        bits = bits_for(data, audit)
        sign_key = signs(data, audit) if endpoint in ("uspto50k_ub","ecreact_1") else None
        collisions = {}
        for width in WIDTHS:
            vector = bits if width == 2048 else bits[:, :128] ^ bits[:, 128:]
            folded = packed_ids(vector)
            relation = np.unique(np.column_stack((folded, exact)), axis=0)
            participating = relation[np.bincount(relation[:, 0])[relation[:, 0]] > 1]
            baseline = archive.loc[archive.endpoint.eq(endpoint) & archive.total_bits.eq(width) & archive.channel_set.eq("D")]
            if endpoint in B_PANELS:
                assert len(baseline) == 1
                assert len(np.unique(participating[:, 1])) == int(baseline.n_collision_classes.iloc[0])
            buckets = Buckets(folded)
            for group in np.unique(participating[:, 0]):
                members = buckets.members(group)
                if endpoint == "ecreact_1":
                    if len(np.unique(sign_key[members])) == 1:
                        continue
                key = tuple(sorted(set(exact[members])))
                if key not in collisions:
                    collisions[key] = [members, []]
                np.testing.assert_array_equal(members, collisions[key][0])
                collisions[key][1].append(width)
        for i, (_, (members, widths)) in enumerate(sorted(collisions.items()), 1):
            pair = f"B_{endpoint}_{i:02d}"
            assert len(set(exact[members])) == 2
            candidates = []
            for a, b in combinations(members, 2):
                if exact[a] == exact[b]:
                    continue
                if endpoint == "ecreact_1" and sign_key[a] == sign_key[b]:
                    continue
                order = sorted([a,b], key=lambda k: (*chemistry(data["texts"][k])[:1],data["ids"][k],data["pos"][k]))
                first, second = (from_source(data, k, pair, side, ec) for side,k in enumerate(order,1))
                present = [x["scaffold"] not in used for x in (first, second)]
                distinct = first["scaffold"] != second["scaffold"]
                count = sum(present) if distinct else int(any(present))
                rank = (-count, first["n_heavy_atoms"]+second["n_heavy_atoms"],
                        first["r_id"], second["r_id"], first["source_position"], second["source_position"])
                candidates.append((rank, order, (first, second)))
            _, indices, sides = min(candidates,key=lambda q:q[0])
            delta_and_side = [exact_delta(data, k, repaired) for k in indices]
            a, b = [q[0] for q in delta_and_side]
            assert a != b
            stage = "support" if sign(a) == sign(b) else "folding"
            proof = {}
            for width in widths:
                vector = bits if width == 2048 else bits[:,:128]^bits[:,128:]
                np.testing.assert_array_equal(vector[indices[0]], vector[indices[1]])
                for k, (_, r, p, *_rest) in zip(indices, delta_and_side, strict=True):
                    np.testing.assert_array_equal(NATIVE._packed(r,p,"parity",width,"delta"),vector[k])
                proof[str(width)] = bit_witness(a,b,width)
            assert any(proof.values()) == (stage == "folding")
            tokens = [[f"{k:016x}",a.get(k,0),b.get(k,0)] for k in sorted(a.keys() | b.keys())
                      if a.get(k,0) != b.get(k,0)]
            inventory.append(dict(endpoint=endpoint,pair_id=pair,eligible_row_pairs=len(candidates),
                                  bucket_rows=len(members),exact_D_classes=2,stage=stage,widths=widths,
                                  left_r_id=sides[0]["r_id"],right_r_id=sides[1]["r_id"],
                                  left_label=sides[0]["label"],right_label=sides[1]["label"],
                                  differing_signed_counts=tokens,bit_witnesses=proof))
            for row in sides:
                row.update(collision_stage=stage, bit_widths=jtext(widths),
                           differing_signed_counts_left_right=jtext(tokens), folded_bit_keys=jtext(proof))
                if row["scaffold"] in used:
                    withheld.append({k:v for k,v in row.items() if k not in ("rxn","mapped_rxn","mapping_origin")} |
                                    {"part":"B", "reason":"scaffold already used by the paired or an earlier qualifying row"})
                else:
                    used.add(row["scaffold"])
                    rows.append(row)
    return rows, inventory


def reagent_examples(audit, used):
    root = BASE/"reagent_prediction_v1"
    plan_path = root/"plan.json"
    plan_hash = read(audit.use(root/"plan_seal.json"))["sha256"]
    plan = read(audit.use(plan_path, plan_hash))
    split, = (r for r in plan["splits"] if r["name"] == "native_seed42")
    select = audit.use(split["path"], plan["pinned_files"][split["path"]])
    with np.load(select, allow_pickle=False) as z:
        test = z["test"]
    syn = root/"models/synrfp_delta_r2b2048__native_seed42"
    src_public = WORKSPACE/"SynRFP/Experiments/65_default_optimal/publication_metadata/default_optimal_v1/reagent/cells.csv"
    cells = pd.read_csv(audit.use(src_public))
    selected = cells.loc[cells.method.eq("synrfp_default") & cells.split.eq("native_seed42")]
    assert selected.source_result_path.tolist() == [str(syn/"result.json")]
    result = read(audit.use(syn/"result.json", selected.source_result_sha256.iloc[0]))
    assert result["plan_sha256"] == plan_hash
    assert result["split"] == split and result["method"] == "synrfp_delta_r2b2048"
    audit.use(syn/"rows.npz", result["artifacts"]["rows.npz"])
    assert result["artifacts"]["rows.npz"] == audit.pins[str(select)]
    with np.load(syn/"rows.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["test"], test)
    vocab_path = audit.use(syn/"vocabulary.json", result["artifacts"]["vocabulary.json"])
    vocabulary = np.asarray(read(vocab_path))
    p = audit.use(syn/"test_probability.npy", result["artifacts"]["test_probability.npy"])
    scores = np.load(p, mmap_mode="r", allow_pickle=False)
    assert scores.shape == (len(test),len(vocabulary)) and scores.dtype == np.float32
    assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
    syn_prediction = [set(vocabulary[row >= result["threshold"]]) for row in scores]

    t5 = BASE/"specialist_references_v1/t5chem"
    plan_t5 = t5/"plan.json"
    splan = read(audit.use(plan_t5, read(audit.use(t5/"plan_seal.json"))["sha256"]))
    assert splan["parent_plan_sha256"] == plan_hash
    assert split in splan["splits"] and splan["pinned_files"][str(select)] == audit.pins[str(select)]
    out = t5/"models/native_seed42"
    receipt = read(audit.use(out/"COMPLETE.json"))
    pred_file = out/"predictions.json.gz"
    assert receipt["seed"] == 42 and receipt["protocol"] == "native" and receipt["test_rows"] == len(test)
    with gzip.open(audit.use(pred_file, receipt["artifacts_sha256"][str(pred_file)]), "rt") as stream:
        payload = json.load(stream)
    np.testing.assert_array_equal(payload["rows"], test)
    t5_prediction = [set(row["components"][0]) for row in payload["predictions"]]
    assert len(t5_prediction) == len(test) == 14215

    table_path = audit.use(plan["table"], plan["pinned_files"][plan["table"]])
    f = pd.read_csv(table_path,dtype={"row_id": str},keep_default_na=False)
    refs = read(audit.use(plan["targets"], result["targets_sha256"]))
    np.testing.assert_array_equal(f.source_position, np.arange(len(f)))
    test_rows = f.iloc[test]
    assert test_rows.native_split.eq("test").all() and test_rows.eligible.all()
    reference = [set(refs[i]) for i in test]
    t_correct = np.array([a == b for a,b in zip(t5_prediction, reference, strict=True)])
    s_correct = np.array([a == b for a,b in zip(syn_prediction, reference, strict=True)])
    np.testing.assert_allclose(t_correct.mean(), receipt["metrics"]["exact_set_accuracy"], atol=1e-13, rtol=0)
    np.testing.assert_allclose(s_correct.mean(), result["test"]["exact_set_accuracy"], atol=1e-13, rtol=0)
    bins = {
        "both_exact": t_correct & s_correct,
        "only_t5chem": t_correct & ~s_correct,
        "only_synrfp": ~t_correct & s_correct,
        "neither_exact": ~t_correct & ~s_correct,
    }
    totals = [dict(endpoint="uspto500_reagent",seed=42,outcome=name,n_rows=int(mask.sum()), n_test_rows=len(test))
              for name,mask in bins.items()]
    assert sum(row["n_rows"] for row in totals) == 14215

    rows = []
    for case, quota in (("only_t5chem",4),("only_synrfp",2)):
        candidates = []
        for i in np.flatnonzero(bins[case]):
            item = test_rows.iloc[i]
            position = item.source_position
            row = row_values("uspto500_reagent", "", 1, item.row_id, item.rxn, "", position)
            row.update(outcome=case, label=jtext(sorted(reference[i])), true_reagents=jtext(sorted(reference[i])),
                       t5chem_predicted_reagents=jtext(sorted(t5_prediction[i])),
                       synrfp_predicted_reagents=jtext(sorted(syn_prediction[i])))
            candidates.append(row)
        chosen = 0
        for row in sorted(candidates,key=lambda r:(r["n_heavy_atoms"],r["r_id"],r["source_position"])):
            if row["scaffold"] in used:
                continue
            chosen += 1
            row["pair_id"] = f"D_{case}_{chosen:02d}"
            used.add(row["scaffold"])
            rows.append(row)
            if chosen==quota:
                break
        assert chosen == quota
    return rows, totals


def attach_maps(audit, rows):
    """Attach only archived maps for the same chemically configured sides."""
    sources = {"tpl_b":"tpl_b","ecreact_1":"ecreact","uspto50k_ub":"uspto_50k_u"}
    root = BASE/"syncluster_main12_v1/inputs/sources"
    groups_by_source = defaultdict(list)
    for r in rows:
        if r["endpoint"] in sources:
            groups_by_source[sources[r["endpoint"]]].append(r)
    inventory = {}
    for source, cases in groups_by_source.items():
        wanted = defaultdict(list)
        for row in cases:
            wanted[row["source_position"]].append(row)
        ready = read(audit.use(root/f"{source}.READY.json"))
        ledger = root/f"{source}.rows.jsonl"
        found = set()
        with audit.use(ledger, ready["artifacts_sha256"][str(ledger)]).open() as stream:
            for line in stream:
                saved = json.loads(line)
                position = int(saved["source_position"])
                if position not in wanted:
                    continue
                assert position not in found
                found.add(position)
                for row in wanted[position]:
                    assert saved["row_id"] == row["r_id"] and saved["reaction"] == row["rxn"]
                    text = saved["mapped_reaction"]
                    if saved["status"] == "ok" and text and canonical_sides(text) == canonical_sides(row["rxn"]):
                        row.update(mapped_rxn=text,mapping_origin=saved["mapping_origin"])
                if len(found) == len(wanted):
                    break
        assert len(found) == len(wanted)
        inventory[source] = sum(bool(r["mapped_rxn"]) for r in cases)

    reagent = [r for r in rows if r["endpoint"]=="uspto500_reagent"]
    if reagent:
        dir23 = BASE/"reagent_prediction_v1"
        plan = read(dir23/"plan.json")
        raw = dir23/"raw/uspto_500.csv.gz"
        table = pd.read_csv(audit.use(raw, plan["source_sha256"]), usecols=["r_id","rxn"],keep_default_na=False)
        for row in reagent:
            original = table.iloc[row["source_position"]]
            assert original.r_id == row["r_id"]
            if re.search(r":\d+\]", original.rxn) and canonical_sides(original.rxn) == canonical_sides(row["rxn"]):
                row.update(mapped_rxn=original.rxn,mapping_origin="SynRXN source")
        inventory["uspto500_reagent"] = sum(bool(r["mapped_rxn"]) for r in reagent)
    return inventory


def main():
    audit = InputAudit()
    repaired = init_native(audit)
    ec = load_ec_labels(audit)

    @lru_cache(maxsize=4)
    def source(endpoint):
        return frozen_source(audit,endpoint)

    OUTPUT.mkdir(parents=True,exist_ok=True)
    used, withheld = set(), []
    c, reversible = reverse_equal(audit, used, repaired, ec, source, withheld)
    print("C",reversible,flush=True)
    b, collisions = fold_cases(audit, used, repaired, ec, source, withheld)
    print("B",[(x["endpoint"],x["stage"],x["widths"]) for x in collisions],flush=True)
    a, magnitudes = magnitude_pairs(audit, used, repaired, ec, source)
    print("A",magnitudes,flush=True)
    d, outcomes = reagent_examples(audit, used)
    print("D",[(x["outcome"],x["n_rows"]) for x in outcomes],flush=True)
    all_rows = a+b+c+d
    assert len({r["scaffold"] for r in all_rows}) == len(all_rows)
    mapping = attach_maps(audit, all_rows)
    for filename, rows in (("count_magnitude_pairs",a),("fold_collision_pairs",b),
                           ("reverse_equal_reactions",c),("reagent_examples",d),("reagent_outcomes",outcomes)):
        export(OUTPUT/f"{filename}.csv",rows)
    write(OUTPUT/"witness_inventory.json",dict(count_magnitude=magnitudes, fold_collisions=collisions,
          reverse_equal=reversible, reagent_outcomes=outcomes, source_mapping_rows=mapping,
          withheld_by_scaffold_rule=withheld))
    fmt_a = " and ".join(f"{x['exported_pairs']} of {x['eligible_original_row_pairs']:,} eligible row pairs for {k}"
                         for k,x in magnitudes.items())
    fmt_b = "; ".join(f"{x['endpoint']} at {', '.join(str(w) for w in x['widths'])} bits ({x['stage']})" for x in collisions)
    selected_b = len({r["pair_id"] for r in b})
    note = f"""# Reaction witnesses for the supplementary analyses

**A. Count magnitudes.** {fmt_a}. Both members of each pair have the same signed token support and different observed labels. The differing count triples and rooted WL environments are included. The sign-only replay added {magnitudes['ecreact_1']['added_forced_errors']} forced errors on ECReact L1 and {magnitudes['tpl_b']['added_forced_errors']} on TPL B.

**B. Compression.** The two exact D classes in the prior six-endpoint summary are the two sides of one USPTO50K UB collision. The genuine ECReact L1 fold merge is a second case. The cases are {fmt_b}. The audit contains both IDs, counts, labels and differing-bit token lists for both cases. The drawing CSV covers {selected_b} cases. A partner sharing a scaffold is recorded in the inventory with its ID.

**C. Reversal.** {reversible['eligible_reactions']} of the 8,000 Request 86 reactions had identical valid vectors in the two directions. The drawing CSV contains {reversible['displayed_reactions']} unique scaffolds. The inventory records the qualifying rows omitted by scaffold deduplication. The cause field uses canonical molecular components with and without stereo while preserving multiplicities.

**D. Reagent prediction.** Seeds and 14,215 native test rows match both released arms. T5Chem uses its first saved candidate and SynRFP uses the saved validation threshold. The four-cell table counts exact agreement with the full reference set. The drawing CSV shows four T5Chem-only and two SynRFP-only successes.

Reversal and fold cases take priority for the shared scaffold inventory, followed by magnitude pairs and reagent examples. Single rows rank by total heavy atoms, row ID, and source position. Magnitude pairs rank by combined heavy atoms, ordered member ranks, and endpoint. For fold cases, the rule first maximizes the number of new drawable scaffolds and then uses the same size and identity tie-breaks. The scaffold is the Bemis–Murcko scaffold of the largest molecular component across the reaction, with the Request 75 canonical largest-component identity convention for acyclic molecules. Mapped SMILES come from local archived maps that preserve the configured input and product component identities.
"""
    (OUTPUT/"note.md").write_text(note)
    audit.use(BRANCH75/"common.py")
    audit.use(BRANCH75/"drfp_cases.py")
    audit.use(HERE/"REQUEST.md")
    write(OUTPUT/"COMPLETE.json",dict(status="complete",request_revision="9c8046e4d8cf214243eef24136daeb8b48a77f92",
          source_commits=dict(request74="c2f046a9724a70d4821e16d5a6b27b546003f8fe",request75="9e2c2de9c38d7ad342bbd86e744d5684b3024e2f",
                              request78_ec_labels="e791ffd5",request85="5bcedecdfad80a1c9ce552422f4846b7783e1c84",
                              request86="3fdd44463fe4cdec76a92e62ecc6245474c650f6",
                              request86_error_floor="d21452ccbfbd47acc036d5cc4565cd55c62bba69"),
          input_sha256=audit.pins.copy(), code_sha256={p.name:audit.use(p) and audit.pins[str(p)] for p in HERE.glob("*.py")},
          output_sha256={p.name:audit.use(p) and audit.pins[str(p)] for p in OUTPUT.iterdir() if p.name!="COMPLETE.json"},
          packages={p:version(p) for p in ("numpy","pandas","rdkit","synkit","drfp")}, no_model_fits=True,
          exported_rows=dict(A=len(a),B=len(b),C=len(c),D=len(d))))
    print("Published aggregate evidence and selected witnesses",mapping,flush=True)


if __name__ == "__main__":
    main()
