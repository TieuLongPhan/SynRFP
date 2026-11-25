"""Check and seal the selected chemistry-case export."""
from __future__ import annotations

from collections import Counter
import csv
from importlib.metadata import version
import json
import subprocess

import numpy as np

from common import HERE, OUTPUT, LOCAL, WORKSPACE, CEILING, InputAudit, read, write


def rows(name):
    with (OUTPUT/name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def distinct_scaffolds(records, key):
    labels={}
    for row in records:
        labels.setdefault(row["scaffold"],set()).add(row[key])
    assert all(len(values)==1 for values in labels.values())


def verify():
    a,b,c,d,e,f=(rows(f"case_{name}.csv") for name in "abcdef")
    assert len(a)==60 and len(b)==72
    assert set(Counter(r["endpoint"] for r in a).values())=={12}
    assert set(Counter(r["endpoint"] for r in b).values())=={12}
    distinct_scaffolds(a,"case_id")
    distinct_scaffolds(b,"case_id")
    assert all(r["pred_D"]!=r["true_label"]==r["pred_DU"] and
               r["collision_label"]!=r["true_label"] and r["collision_r_id"]
               and int(r["n_D_wrong_DU_right"])*2>int(r["n_partitions"]) for r in a)
    assert all(r["pred_DU"]!=r["true_label"]==r["pred_D"] and
               int(r["n_D_right_DU_wrong"])*2>int(r["n_partitions"]) for r in b)
    assert all(int(r["n_classes_merged_1024"])-int(r["n_classes_merged_2048"])==
               int(r["n_classes_merged_only_1024"]) for r in b)
    fold=rows("fold_collision_summary.csv")
    assert len({(r["endpoint"],r["channel_set"],r["total_bits"]) for r in fold})==len(fold)==36
    for r in fold:
        np.testing.assert_allclose(float(r["collision_class_share"]),
                                   int(r["n_collision_classes"])/int(r["n_exact_channel_classes"]))
    assert set(Counter(r["pair_id"] for r in c).values())=={2}
    distinct_scaffolds(c,"pair_id")
    for r in c:
        assert (r["drfp_equal"]=="True")!=(r["D_equal"]=="True")
        assert (r["direction"]=="drfp_equal_D_different")==(r["drfp_equal"]=="True")
        assert bool(json.loads(r["differing_tokens"]))==(r["D_equal"]=="False")
    assert len(d)==4736 and set(Counter(r["recipe_id"] for r in d).values())=={2}
    zero={}
    for r in d:
        np.testing.assert_allclose(float(r["pred_zero_gap_arm"]),
                                   zero.setdefault(r["recipe_id"],float(r["pred_zero_gap_arm"])),atol=1e-5,rtol=0)
    distinct_scaffolds(e,"case_id")
    assert len(e)==120 and set(Counter(r["case_id"] for r in e).values())=={20}
    for case in {r["case_id"] for r in e}:
        for k in ("rank_D","rank_DU"):
            assert {int(r[k]) for r in e if r["case_id"]==case}==set(range(1,21))
    assert len(f)==20 and set(Counter((r["dataset"],r["direction"]) for r in f).values())=={5}
    for dataset in {r["dataset"] for r in f}:
        sub=[r for r in f if r["dataset"]==dataset]
        distinct_scaffolds(sub,"case_id")
        assert all(int(r["paired_seed_votes"])>=2 for r in sub)
        if dataset=="enzymemap":
            assert all((r["pred_DU_ec"]==r["true_ec_label"])==(r["direction"]=="gain")
                       and (r["pred_D_ec"]==r["true_ec_label"])==(r["direction"]=="loss") for r in sub)
    return dict(zip("ABCDEF",map(len,(a,b,c,d,e,f)),strict=True))


def readme(counted, sourced):
    a,b=[list(r for r in sourced["classification_COMPLETE"]["counts"] if r["case"]==case) for case in "AB"]
    c=sourced["C_COMPLETE"]["counts"]
    d,e,f=[sourced[k+"_COMPLETE"] for k in "DEF"]
    lines=["# Chemistry cases for the Supplementary Information","",
           "[Selection rules](selection_rules.md) give the caps, tie breaks and scaffold checks.",
           "[AUDIT.json](AUDIT.json) records input hashes, versions and candidate counts.",
           "", "## A and B. Classification", "",
           "| Case | Endpoint | Majority-vote qualified | Encoded | With required A witness | Exported |",
           "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in a+b:
        witness=str(r["witness_qualified"]) if r["case"]=="A" else "—"
        lines.append(f"| {r['case']} | {r['endpoint']} | {r['vote_qualified']:,} | {r['encoded_vote_qualified']:,} | {witness} | {r['exported']} |")
    lines.extend([
        "", f"[Case A](case_a.csv) contains {counted['A']} rescues with training witnesses. "
            f"[Case B](case_b.csv) contains {counted['B']} failures.",
        f"The 1,024-only D screen found {sum(r['qualified_with_1024_only_collision'] for r in b)} width-specific collisions "
        f"among {sum(r['encoded_vote_qualified'] for r in b):,} encoded failures.",
        "See the [fold collision rates](fold_collision_summary.csv).",
        "", "## C. DRFP and signed-count disagreements","",
        "| Dataset | Direction | Eligible unordered pairs | Representative buckets | Exported pairs |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for r in c:
        lines.append(f"| {r['dataset']} | {r['direction']} | {r['eligible_pairs']:,} | {r['eligible_buckets']:,} | {r['exported_pairs']} |")
    same_bits = [r for r in rows("case_c.csv") if r["side"]=="1" and r["drfp_equal"]=="True"]
    same_fragments = sum(r["drfp_fragments_equal"]=="True" for r in same_bits)
    lines.extend([
        "", "[Case C](case_c.csv) contains both rows of every pair, decoded DRFP fragments,",
        "and rooted WL environments for the differing signed tokens.",
        f"Among the {len(same_bits)} equal-DRFP/different-D pairs, {same_fragments} have identical "
        f"pre-fold DRFP fragment sets and {len(same_bits)-same_fragments} share only the folded bits.", "",
        "## D–F. Yield, condition ranking, and external tasks","",
        f"- [Case D](case_d.csv) contains {d['assay_pairs']:,} recipe-matched pairs "
        f"across {d['holdouts']} E3 holdouts and {counted['D']:,} assay appearances.",
        f"- [Case E](case_e.csv) contains all {e['counts']['eligible_gain']} alcohol gains and "
        f"all {e['counts']['eligible_loss']} losses among the {e['eligible_substrates']} "
        "substrates with saved scores. Each selected substrate has all 20 condition rankings.",
        f"- [Case F](case_f.csv) contains five gains and five losses per dataset. EnzymeMap had "
        f"{f['enzymemap']['counts']['eligible_gain']:,} eligible gains and "
        f"{f['enzymemap']['counts']['eligible_loss']:,} losses. QuantumPioneer had "
        f"{f['quantumpioneer']['counts']['eligible_gain']:,} eligible gains and "
        f"{f['quantumpioneer']['counts']['eligible_loss']:,} losses.",
        "", "See the [alcohol stereochemistry probe](stereo_probe.md) for Request 74's bit-vector check.",
    ])
    (OUTPUT/"README.md").write_text("\n".join(lines)+"\n")


def main():
    counted=verify()
    names=("classification_COMPLETE","C_COMPLETE","D_COMPLETE","E_COMPLETE","F_COMPLETE")
    sourced={k:read(LOCAL/(k+".json")) for k in names}
    sources={}
    for value in sourced.values():
        sources.update(value.get("inputs_sha256",{}))
    audit=InputAudit()
    for marker in ("ENCODING_PLAN.json","ENCODED.json"):
        audit.use(CEILING/marker,sources[str(CEILING/marker)])
    root=WORKSPACE/"SynRFP-delta-ceiling/Experiments/74_delta_ceiling/publication_metadata/delta_ceiling_v1"
    complete=read(audit.use(root/"COMPLETE.json"))
    stereo=read(audit.use(root/"stereo_feasibility.json",complete["files_sha256"]["stereo_feasibility.json"]))
    assert stereo["all_four_stereo_aware_fingerprints_distinct"] and not stereo["all_four_ordinary_fingerprints_distinct"]
    (OUTPUT/"stereo_probe.md").write_text(
        "# Alcohol stereochemistry probe\n\n"
        "Request 74's rooted SynKit stereo labels separated both alcohol pairs: "
        "the Boc hydroxyproline methyl esters and the phthalimidocyclohexanols. "
        "The ordinary WL/r2 delta fingerprints matched within both pairs. "
        "The stereo-aware fingerprints were distinct for all four substrates "
        "and invariant under the tested atom renumbering. This is a four-compound "
        "fingerprint-separation check.\n"
    )
    readme(counted,sourced)
    pinned_input=sources|audit.pins
    scripts=[audit.use(p) for p in sorted(HERE.glob("*.py"))]
    write(OUTPUT/"AUDIT.json",{
        "status":"complete","request_commit":"3b63d0d7","count_class_release":"c2f046a9",
        "source_counts":{key:{k:v for k,v in val.items() if k!="inputs_sha256"} for key,val in sourced.items()},
        "code_sha256":{p.name:audit.pins[str(p)] for p in scripts},
        "inputs_sha256":pinned_input,
        "package_versions":{k:version(k) for k in ("numpy","pandas","rdkit","drfp","synkit","scikit-learn")},
        "producer_checkout":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "new_fits":0,"verified_csv_rows":counted,
    })
    files=[audit.use(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name!="COMPLETE.json"]
    write(OUTPUT/"COMPLETE.json",{"status":"complete","exported_csv_rows":counted,
                                  "files_sha256":{p.name:audit.pins[str(p)] for p in files}})
    print(counted,flush=True)


if __name__=="__main__":
    main()
