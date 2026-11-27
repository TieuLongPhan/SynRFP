"""Produce matched feasibility counts and barrier examples for the SI."""
from importlib.metadata import version

from common89 import HERE, OUTPUT, REQ78, REQ88, Pins, digest, export, write
import barrier89
import feasibility89


def main():
    pins = Pins()
    used = set()
    feasibility, outcomes = feasibility89.compare(pins,used)
    print("Feasibility",outcomes,flush=True)
    barriers, inventory = barrier89.compare(pins,used)
    print("Barriers",inventory,flush=True)
    assert len(used) == len(feasibility)+len(barriers)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for name,rows in (("feasibility_outcomes",outcomes),("feasibility_examples",feasibility),("barrier_examples",barriers)):
        export(OUTPUT/f"{name}.csv",rows)
    descriptions = []
    for dataset in ("gdb7","rgd1"):
        cases=[r for r in barriers if r["endpoint"]==dataset and r["direction"]=="fp_larger_error"]
        ranges = [r["reactant_heavy_atoms"] for r in cases]
        elements = sorted(set.union(*(set(r["reactant_elements"]+r["product_elements"]) for r in cases)))
        changes = sum(r["reactant_ring_sizes"]!=r["product_ring_sizes"] for r in cases)
        gaps = [r["synrfp_minus_dmpnn_abs_error"] for r in cases]
        selected_exact = [r["training_exact_signed_count_neighbours"] for r in cases]
        descriptions.append(f"**{dataset.upper()}.** The three selected SynRFP losses have "
          f"{min(ranges)}–{max(ranges)} reactant heavy atoms and contain {', '.join(elements)}. "
          f"The ring-size lists differ from reactants to products in {changes}/3, and the absolute-error "
          f"gaps range from {min(gaps):.2f} to {max(gaps):.2f} in the stored benchmark units. "
          f"The counts of actual training rows sharing their exact signed WL radius-2 counts are "
          f"{', '.join(str(x) for x in selected_exact)}.")
    largest_rgd1 = next(r for r in barriers if r["pair_id"] == "B_rgd1_fp_larger_error_01")
    if largest_rgd1["rxn"] == "CNN>>CNN" and largest_rgd1["n_signed_wl_tokens"] == 0:
        descriptions[-1] += (
            f" The largest loss, `{largest_rgd1['r_id']}`, has identical unmapped sides "
            "and an empty signed delta. The saved mapped representation places hydrogen "
            "atom labels 4 and 6 on opposite nitrogen atoms across the two sides."
        )
    note = f"""# Feasibility and activation-barrier examples

**Feasibility.** Both seed-1 models scored the same fixed 751 expert rows, including 341 feasible reactions. Thresholds come from the archived calibration decisions and were independently recomputed from the saved calibration scores. The two disagreement pools contain only expert-infeasible rows for the figures. Within each pool, rows with a recorded failure reason rank first, followed by heavy-atom count, stable ID, and source position. The counts table covers both expert classes.

**Barrier comparison.** Every GDB7 and RGD1 row appears once in the paired grouped-fold predictions. Candidate ranking uses the difference in absolute errors, with fewer heavy atoms, row ID, and source position as tie-breaks. Each endpoint contributes its three largest scaffold-eligible SynRFP losses and one largest D-MPNN loss. Counts refer to the actual fitting rows, excluding validation. The released delta sketch narrows the search and every matched-sketch candidate is checked with exact signed WL radius-2 counts. GDB7 targets are in kcal/mol. RGD1 uses the retained release target values for both models.

{descriptions[0]}

{descriptions[1]}

Scaffold identity follows Request 88 across this export, including the canonical largest-component fallback for acyclic reactions. Eligible outcome and barrier pools are recorded in the counts table and manifest. These are selected examples from saved predictions. Chemical descriptions are limited to the observed structures.
"""
    (OUTPUT/"note.md").write_text(note)
    for dep in (REQ78/"barrier78.py",REQ78/"common78.py",REQ88/"common88.py",REQ88/"select88.py"):
        pins.use(dep)
    write(OUTPUT/"COMPLETE.json",dict(status="complete",request_revision="e86bc6047116c68756aab60c446eb362bd81047f",
          reference_commits=dict(request88="c6ea36bfa0f930536fcab344f5a9492d170e0938",
                                 gdb7_grouped="445dcf5119cf4580753eb57d32c207362bb2a301"),
          feasibility_test_rows=751,feasibility_figures=len(feasibility),barrier_inventory=inventory,
          barrier_figures=len(barriers),scaffold_count=len(used),no_new_fits=True,
          input_sha256=pins.hashes,code_sha256={p.name:digest(p) for p in sorted(HERE.glob("*.py"))},
          output_sha256={p.name:digest(p) for p in OUTPUT.iterdir() if p.name!="COMPLETE.json"},
          packages={p:version(p) for p in ("numpy","pandas","scipy","synkit","rdkit")}))


if __name__=="__main__":
    main()
