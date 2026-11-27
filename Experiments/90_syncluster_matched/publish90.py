"""Verify completed cell receipts and publish metrics without model artifacts."""
import csv
from importlib.metadata import version
import statistics
import time

import numpy as np

from common90 import BASELINES, COUNT, HERE, METHOD, N, OUTPUT, RELEASE, ROOT, SOURCE, UPSTREAM, read, sha, write
from kmeans90 import score

METRICS=("ami","ari","nmi","purity","actual_clusters","largest_cluster_fraction",
         "singleton_reaction_fraction","singleton_cluster_fraction")


def verified(path, expected):
    if sha(path)!=expected:
        raise ValueError(f"Changed archived input: {path}")
    return path


def load(path):
    with np.load(path,allow_pickle=False) as v:
        return dict(v)


def csv_write(path,records):
    columns=list(dict.fromkeys(key for r in records for key in r))
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns,lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def reference_cells(y, cohort, p):
    p_ref=RELEASE/"comparison_cells.csv"
    with verified(p_ref,p["input_sha256"][str(p_ref)]).open(newline="") as f:
        recorded=list(csv.DictReader(f))
    native=read(SOURCE/"NATIVE_COMPLETE.json")
    rows=[]
    for baseline, found in zip(("raw","residual",*range(10)),recorded,strict=True):
        if baseline in ("raw","residual"):
            pred=cohort[baseline]
            row=dict(method=found["method"],algorithm="Butina" if baseline=="raw" else "Butina + residual merge",
                     distance_cutoff=.6,similarity_threshold=.4,seed="native_order",n=N,source="archived",
                     stage="existing",runtime_seconds=native["partition"]["seconds"])
        else:
            folder=SOURCE/"matched_synrfp"
            q=folder/f"seed_{baseline}.npz"
            receipt=read(q.with_suffix(".json"))
            saved=load(verified(q,receipt["artifact_sha256"]))
            np.testing.assert_array_equal(saved["source_positions"],cohort["source_positions"])
            np.testing.assert_array_equal(saved["row_ids"],cohort["row_ids"])
            pred=saved["labels"]
            row=dict(method=METHOD,algorithm="MiniBatchKMeans",requested_k=147,seed=baseline,n=N,
                     source="archived",stage="existing", runtime_seconds=receipt["seconds"])
        values=score(y,pred)
        for metric,v in values.items():
            if metric in found:
                np.testing.assert_allclose(v,float(found[metric]),atol=1e-12,rtol=0)
        rows.append(row|values)
    return rows


def kmeans_cells(y, prep_sha):
    folder=ROOT/"kmeans"
    rows=[]
    expected={(METHOD,77),(METHOD,378),(COUNT,147),(COUNT,378),*((name,378) for name in BASELINES)}
    seconds={}
    for label in ("native","baselines"):
        receipt=read(ROOT/f"kmeans_{label}_COMPLETE.json")
        assert receipt["n_fits"] == (40 if label=="native" else 50)
        assert receipt["prepared_sha256"]==prep_sha
        seconds[label]=receipt["runtime_seconds"]
        for name,digest in receipt["artifacts"].items():
            record=read(verified(folder/name,digest))
            filename=(folder/name).with_suffix(".npz")
            labels=load(verified(filename,record["partition_sha256"]))["labels"]
            assert score(y,labels)=={k:record[k] for k in METRICS}
            record.setdefault("initialization_arithmetic","native")
            record["stage"]="matched_count"
            rows.append(record)
    ids={(r["method"],r["requested_k"],r["seed"]) for r in rows}
    assert len(rows)==len(ids)==90
    assert ids=={(method,k,s) for method,k in expected for s in range(10)}
    return rows,seconds


def butina_cells(y):
    sweep=read(ROOT/"SWEEP_COMPLETE.json")
    checks={}
    native_receipts={}
    exact=read(ROOT/"KMEANS_INTEGER_REPLAY.json")
    assert len(exact["controls"])==4 and exact["source_sha256"]==sha(HERE/"exact_init90.py")
    upstream_sha=sha(UPSTREAM/"clustering/scripts/03_cluster.py")
    for short in ("synrfp","drfp"):
        f=ROOT/f"native_butina_{short}.npz"
        rec=read(f.with_suffix(".json"))
        assert rec["cutoff"]==.6 and rec["n"]==N and rec["rdkit"]=="2021.09.5"
        assert rec["upstream_source_sha256"]==upstream_sha
        partition=load(verified(f,rec["sha256"]))
        centers,labels=partition["centers"],partition["labels"]
        assert len(centers)==len(np.unique(labels))==rec["actual_clusters"]
        np.testing.assert_array_equal(labels[centers],np.arange(len(centers)))
        checks[short]=rec|partition
        native_receipts[short]=rec
    rows=[]
    for name,digest in sweep["artifact_sha256"].items():
        folder=ROOT/"sweep"
        record=read(verified(folder/name,digest))
        array=load(verified((folder/name).with_suffix(".npz"),record["partition_sha256"]))
        assert score(y,array["labels"])=={k:record[k] for k in METRICS}
        if record["distance_cutoff"]==.6:
            np.testing.assert_array_equal(array["labels"],checks["synrfp"]["labels"])
            np.testing.assert_array_equal(array["centers"],checks["synrfp"]["centers"])
            record["native_full_partition_replay_exact"]=True
        record["closest_to_378_on_coarse_grid"]=record["distance_cutoff"]==sweep["coarse_distance_cutoff"]
        record["closest_to_378_after_refinement"]=record["distance_cutoff"]==sweep["selected_distance_cutoff"]
        rows.append(record)
    d=checks["drfp"]
    rows.append(dict(method="drfp",algorithm="Butina",distance_cutoff=.6,similarity_threshold=.4,
                     seed="native_order",n=N,source="new",stage="native_cutoff",
                     partition_sha256=d["sha256"],runtime_seconds=d["runtime_seconds"])|score(y,d["labels"]))
    assert sum(r.get("native_full_partition_replay_exact",False) for r in rows)==1
    return rows,dict(sweep=sweep,integer_controls=exact,native_butina=native_receipts)


def summarize(rows):
    groups={}
    for row in rows:
        k=(row["method"],row["algorithm"],row.get("requested_k"),row.get("distance_cutoff"))
        groups.setdefault(k,[]).append(row)
    result=[]
    for (method,algorithm,k,cutoff),cells in groups.items():
        record=dict(method=method,algorithm=algorithm,requested_k=k,distance_cutoff=cutoff,
                    similarity_threshold=None if cutoff is None else 1-cutoff,n=N,
                    n_runs=len(cells),source=cells[0]["source"],stage=cells[0]["stage"],
                    closest_to_378_on_coarse_grid=cells[0].get("closest_to_378_on_coarse_grid",False),
                    closest_to_378_after_refinement=cells[0].get("closest_to_378_after_refinement",False))
        for metric in (*METRICS,"runtime_seconds"):
            values=[r[metric] for r in cells]
            record[metric+"_mean"]=statistics.mean(values)
            record[metric+"_sample_sd"]=statistics.stdev(values) if len(values)>1 else None
        result.append(record)
    return result


def note(summary,diagnostic,p,wall,elapsed):
    table={(r["method"],r["algorithm"],r["requested_k"],r["distance_cutoff"]):r for r in summary}
    raw=table["SynCluster_native_raw","Butina",None,.6]
    res=table["SynCluster_native_residual_merged","Butina + residual merge",None,.6]
    k147=table[METHOD,"MiniBatchKMeans",147,None]
    k378=table[METHOD,"MiniBatchKMeans",378,None]
    k77=table[METHOD,"MiniBatchKMeans",77,None]
    syn_b=table[METHOD,"Butina",None,.6]
    fine=table[METHOD,"Butina",None,diagnostic["sweep"]["selected_distance_cutoff"]]
    coarse=table[METHOD,"Butina",None,diagnostic["sweep"]["coarse_distance_cutoff"]]
    template=table[COUNT,"MiniBatchKMeans",378,None]
    template147=table[COUNT,"MiniBatchKMeans",147,None]
    baseline378=[table[fp,"MiniBatchKMeans",378,None] for fp in BASELINES]
    lead=max(baseline378,key=lambda r:r["ami_mean"])
    lines=[
        "# Matched SynCluster comparison on USPTO50K UB",
        "",
        f"The cohort contains the {N:,} admitted SynCluster author records with all ten reference "
        "classes, in archived train-then-validation order. Author positions identify records because "
        f"the archived ID column repeats ({p['distinct_author_ids']:,} distinct IDs). All fingerprint "
        "arms score every admitted row. The five baselines use the released recipes; each reuses "
        "43,040 exact-input rows and encodes the other 341 from the author-normalized inputs. "
        "Each applicable k-means comparison uses seeds 0–9 and the frozen MiniBatchKMeans settings. "
        "The optional template-count k-means comparison preserves all original signed counts in a sparse "
        "2,231-column matrix.",
        "",
        "| Method | Algorithm | Requested k or distance cutoff | Occupied clusters | AMI |",
        "|---|---|---:|---:|---:|",
    ]
    entries=[(raw,"native raw"),(res,"native residual merge"),(k147,"SynRFP standard"),(k77,"SynRFP standard"),
             (k378,"SynRFP standard"),(syn_b,"SynRFP standard"),(coarse,"SynRFP standard · nearest grid count"),
             (fine,"SynRFP standard · nearest refined count"),
             (table["drfp","Butina",None,.6],"DRFP")]
    entries.extend((r,r["method"]) for r in baseline378)
    entries.extend((table[COUNT,"MiniBatchKMeans",k,None],"SynCluster template counts") for k in (147,378))
    for r,label in entries:
        k=r["requested_k"] if r["requested_k"] is not None else f"{r['distance_cutoff']:.3f}"
        value=f"{r['ami_mean']:.4f}"
        if r["ami_sample_sd"] is not None:
            value+=f" ± {r['ami_sample_sd']:.4f}"
        lines.append(f"| {label} | {r['algorithm']} | {k} | {r['actual_clusters_mean']:.1f} | {value} |")
    lines.extend(["",
        f"At k = 378, SynRFP AMI was {k378['ami_mean']:.4f}, compared with "
        f"{k147['ami_mean']:.4f} under its original k = 147 rule and {raw['ami_mean']:.4f} "
        "for archived native SynCluster. Matching the requested cluster count alone left a gap. "
        f"With both representations evaluated by k-means at k = 378, SynCluster template counts "
        f"scored {template['ami_mean']:.4f} with {template['actual_clusters_mean']:.1f} occupied clusters "
        f"on average. At k = 147, both representations occupied all 147 clusters and template counts "
        f"scored {template147['ami_mean']:.4f}. These are comparisons on the author cohort with its "
        "original corrected templates. The upstream template correction included source test inputs "
        "without class labels, as recorded in the archived protocol.",
        "",
        f"Among the other benchmark fingerprints at k = 378, {lead['method']} led with "
        f"mean AMI {lead['ami_mean']:.4f}, compared with {k378['ami_mean']:.4f} for SynRFP.",
        "",
        f"Native Butina uses a distance cutoff of 0.60, equivalent to a similarity threshold of "
        f"0.40 at the stated precision. Both bit-vector arms ran the original RDKit path. "
        f"The full SynRFP replay reproduced its cluster centers and assignments exactly. "
        f"The nearest count on the 0.05 grid was {int(coarse['actual_clusters_mean'])} at "
        f"{coarse['distance_cutoff']:.2f}. Count-only refinement on the intervening interval gave "
        f"{int(fine['actual_clusters_mean'])} at {fine['distance_cutoff']:.3f}. "
        "The refinement used cluster count alone. The CSV reports the complete tested grids and "
        "all requested AMI, ARI, NMI, and occupancy statistics.",
        "",
        "K-means initialization was accelerated for some integer-valued matrices using exact integer "
        "squared distances. The optimizer, update arithmetic and seeds are those of the archived "
        "protocol. Four full-cohort controls verified identical centers and assignments. "
        "Single deterministic runs have empty standard deviations.",
        "",
        f"The run used CPU only and took {elapsed/60:.1f} minutes from the cohort preparation receipt. "
        f"Recorded service wall times were {wall['native']:.1f} s for the "
        f"final primary/template k-means lane, {wall['baselines']:.1f} s for the final baseline lane, "
        f"and {diagnostic['sweep']['runtime_seconds']:.1f} s for the sweep. The two full native "
        f"Butina replays took {diagnostic['native_butina']['synrfp']['runtime_seconds']:.1f} s for SynRFP and "
        f"{diagnostic['native_butina']['drfp']['runtime_seconds']:.1f} s for DRFP. Service timings overlap. "
        "The final k-means lane timings exclude the initial unmodified fits reused from before the acceleration."
    ])
    return "\n".join(lines)+"\n"


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    plan=ROOT/"PREPARED.json"
    p=read(plan)
    cohort=load(verified(ROOT/"cohort.npz",p["local_sha256"]["cohort.npz"]))
    y=cohort["labels"]
    old=reference_cells(y,cohort,p)
    k,wall=kmeans_cells(y,sha(plan))
    b,audit=butina_cells(y)
    rows=old+k+b
    summary=summarize(rows)
    csv_write(OUTPUT/"matched_cells.csv",rows)
    csv_write(OUTPUT/"matched_summary.csv",summary)
    elapsed=time.time()-plan.stat().st_mtime
    (OUTPUT/"note.md").write_text(note(summary,audit,p,wall,elapsed))
    write(OUTPUT/"COMPLETE.json",dict(status="complete",request_revision="397dc658968a000a8e24e77ec4713dd92e454f6a",
          n=N,archived_cells=len(old),new_kmeans_cells=len(k),new_butina_cells=len(b),table_rows=len(rows),
          same_native_author_order=True,preparation=p,prepared_sha256=sha(plan),
          feature_receipts={name:read(ROOT/f"{name}_features.json") for name in BASELINES},
          wall_seconds=wall,elapsed_from_preparation_seconds=elapsed,verification=audit,
          code_sha256={f.name:sha(f) for f in HERE.iterdir() if f.suffix in (".py",".cpp")},
          output_sha256={f.name:sha(f) for f in OUTPUT.iterdir() if f.name!="COMPLETE.json"},
          versions={package:version(package) for package in p["kmeans"]["environment"]}))
    print("Verified",len(rows),"results across",len(summary),"settings",flush=True)


if __name__=="__main__":
    main()
