"""Rank held-out Buchwald–Hartwig recipes and select deterministic examples."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr

from preflight import (HERE, RELEASE, ROOT, METHODS, PAIR, RECIPE, PARTS, digest,
                       read, verify, load_predictions, author_source)

WORK = ROOT.parent/"request92_yield_ood_v1"
OUTPUT = HERE/"publication_metadata/ood_analysis_v1"
BASELINE = "training_recipe_mean"
GLOBAL = "training_global_mean"
Y = "yield_pct"
MIN_GAP = 10.


def data():
    published = RELEASE/"yield/cells.csv"
    receipt = read(RELEASE/"COMPLETE.json")
    records = pd.read_csv(verify(published,receipt["exported_outputs_sha256"]["yield/cells.csv"]),low_memory=False)
    plan = read(verify(ROOT/"plan.json",read(ROOT/"plan_seal.json")["sha256"]))
    for dataset in ("ncs_bh","uspto_yield"):
        table = Path(plan["tables"][dataset]["path"])
        cohort = pd.read_csv(verify(table,plan["pinned_files"][str(table)]),dtype={"row_id":str})
        for split in plan["splits"]:
            key = split["split"]
            if split["dataset"] != dataset:
                continue
            p = Path(split["path"])
            with np.load(verify(p,plan["pinned_files"][str(p)]),allow_pickle=False) as z:
                selectors = {k:z[k] for k in PARTS}
            refs = records[(records.dataset==dataset) & (records.split==key) & records.method.isin(METHODS)]
            assert len(refs)==4 and set(refs.method)==set(METHODS)
            preds = {r.method:load_predictions(r,selectors,cohort)[0] for r in refs.itertuples()}
            yield dataset,key,cohort,selectors,preds


def record_rows(group):
    return group.sort_values(RECIPE,kind="stable").reset_index(drop=True)


def rank_conditions(split, raw, observed, selectors, predictions):
    fit, val, test = (selectors[k] for k in PARTS)
    train = raw.iloc[fit].assign(yield_pct=observed[fit])
    global_mean = float(train[Y].mean())
    recipe_means = train.groupby(RECIPE,dropna=False)[Y].mean()
    indices = raw.iloc[test][PAIR+RECIPE].assign(position=test, yield_pct=observed[test], **predictions)
    seen = pd.MultiIndex.from_frame(raw.iloc[np.r_[fit,val]][PAIR].drop_duplicates())
    pair_index = pd.MultiIndex.from_frame(indices[PAIR])
    indices = indices.loc[~pair_index.isin(seen)]
    rows, examples = [], []
    for pair, group in indices.groupby(PAIR,dropna=False,sort=True):
        values = group.groupby(RECIPE,dropna=False)[[Y,*METHODS]].mean()
        if len(values) < 8:
            continue
        order_keys = values.index
        values[BASELINE] = recipe_means.reindex(order_keys)
        unseen_recipes = int(values[BASELINE].isna().sum())
        values[BASELINE] = values[BASELINE].fillna(global_mean)
        values[GLOBAL] = global_mean
        values = record_rows(values.reset_index())
        y = values[Y].to_numpy()
        top = {}
        for method in (*METHODS,BASELINE,GLOBAL):
            pred = values[method].to_numpy()
            order = np.argsort(-pred,kind="stable")
            top[method] = int(order[0])
            rho = (float(spearmanr(pred,y).statistic) if np.ptp(pred)>0 and np.ptp(y)>0 else None)
            rows.append(dict(split=split,method=method,n_conditions=len(values),
                   n_assay_rows=len(group),n_train_unseen_recipes=unseen_recipes,
                   spearman=rho,top1_success=bool(y[order[0]]>10),
                   top4_success=bool(np.any(y[order[:4]]>10)),any_success=bool(np.any(y>10))))
        first, second = top["synrfp_optimal"],top["drfp"]
        if y[first] > 10 and y[second] <= 10:
            sides = []
            for method, i in (("synrfp_optimal",first),("drfp",second)):
                cond = values.iloc[i]
                matching = np.ones(len(group),dtype=bool)
                for key in RECIPE:
                    if pd.isna(cond[key]):
                        matching &= group[key].isna()
                    else:
                        matching &= group[key].eq(cond[key])
                representative = int(group.loc[matching,"position"].min())
                sides.append((representative, method, cond.to_dict(), int(matching.sum())))
            examples.append((split,pair,sides,len(values)))
    return rows, examples


def structure(raw, pos):
    q = raw.iloc[pos]
    # Source sides consist of separate molecular fields, so dative bond arrows
    # never enter reaction-delimiter parsing.
    left = ".".join(str(q[c]).strip() for c in PAIR+RECIPE)
    right = str(q["Product SMILES"]).strip()
    mols = []
    for text in (left+"."+right).split("."):
        m = Chem.MolFromSmiles(text)
        if m is None:
            return None
        for atom in m.GetAtoms():
            atom.SetAtomMapNum(0)
        mols.append(m)
    most = min(mols,key=lambda m:(-m.GetNumHeavyAtoms(),Chem.MolToSmiles(m,isomericSmiles=False)))
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=most,includeChirality=False)
    return dict(rxn=left+">>"+right, scaffold=scaffold or "acyclic:"+Chem.MolToSmiles(most,isomericSmiles=False),
                n_heavy_atoms=sum(m.GetNumHeavyAtoms() for m in mols),
                aryl_smiles=q[PAIR[0]], amine_smiles=q[PAIR[1]], catalyst_smiles=q[RECIPE[0]],
                base_smiles=q[RECIPE[1]], solvent_smiles=q[RECIPE[2]],
                product_smiles=right)


def select(individual, pair_examples, chemistry):
    used = set()
    rows = []
    inventory = Counter()
    for direction, limit in (("yield_config_gain",4),("yield_config_loss",2)):
        choices = []
        for split,position,preds,y,gap in individual[direction]:
            key = chemistry(position)
            inventory[direction+"_eligible_split_rows"] += 1
            if key is None:
                inventory[direction+"_unparseable"] += 1
                continue
            choices.append((key["n_heavy_atoms"],position,split,preds,y,gap))
        selected = 0
        for _,pos,split,preds,y,gap in sorted(choices,key=lambda a:a[:3]):
            st = chemistry(pos)
            if st["scaffold"] in used:
                continue
            used.add(st["scaffold"])
            selected += 1
            rows.append(dict(endpoint="ncs_bh",pair_id=direction+"_"+str(selected),
                  side="reaction",r_id=str(pos),label=y,source_position=pos,split=split,
                  case_type=direction,observed_yield=y,error_gap=gap,representative_assays=1,
                  condition_count=None,**st,**preds))
            if selected == limit:
                break
        inventory[direction+"_displayed"] = selected
    pairs = []
    for split,pair,sides,n in pair_examples:
        inventory["ranking_eligible_split_pairs"] += 1
        structures = [chemistry(s[0]) for s in sides]
        if any(s is None for s in structures):
            inventory["ranking_unparseable"] += 1
            continue
        pairs.append((sum(s["n_heavy_atoms"] for s in structures),tuple(s[0] for s in sides),split,sides,structures,n))
    selected = 0
    for _,_,split,sides,structures,n in sorted(pairs,key=lambda a:a[:3]):
        scaffolds = {s["scaffold"] for s in structures}
        if used & scaffolds:
            continue
        used.update(scaffolds)
        selected += 1
        for (pos,method,mean,count),st in zip(sides,structures,strict=True):
            rows.append(dict(endpoint="ncs_bh",pair_id=f"ranking_gain_{selected}",
                        side=method+"_top1",r_id=str(pos),label=mean[Y],source_position=pos,split=split,
                        case_type="ranking_gain",observed_yield=mean[Y],error_gap=None,
                        representative_assays=count,condition_count=n,**st,**{m:mean[m] for m in METHODS}))
        if selected == 2:
            break
    inventory["ranking_displayed"] = selected
    return rows,dict(inventory)


def main():
    from functools import lru_cache

    RDLogger.DisableLog("rdApp.*")
    all_ranks, rank_examples = [], []
    individual = {"yield_config_gain":[],"yield_config_loss":[]}
    raw = None
    for dataset,split,cohort,selectors,pred in data():
        if dataset != "ncs_bh":
            break
        if raw is None:
            raw = author_source(cohort)
        y = cohort[Y].to_numpy()
        rank, examples = rank_conditions(split,raw,y,selectors,pred)
        all_ranks.extend(rank)
        rank_examples.extend(examples)
        test = selectors["test"]
        baseline = np.minimum(np.abs(y[test]-pred["drfp"]),np.abs(y[test]-pred["rdkit_struct"]))
        gap = baseline - np.abs(y[test]-pred["synrfp_optimal"])
        for direction, mask in (("yield_config_gain",gap>=MIN_GAP),("yield_config_loss",gap<=-MIN_GAP)):
            for i in np.flatnonzero(mask):
                individual[direction].append((split,int(test[i]),{m:float(pred[m][i]) for m in METHODS},
                                              float(y[test[i]]),float(gap[i])))
    table = pd.DataFrame(all_ranks)
    pre = pd.read_csv(HERE/"publication_metadata/preflight_v1/condition_coverage.csv")
    expected = pre.set_index("split").pairs_absent_from_train_and_validation
    matches = table.groupby(["split","method"]).size().unstack(fill_value=0).reindex(expected.index,fill_value=0)
    assert matches.columns.size==6 and (matches.eq(expected,axis=0)).all().all() and expected.sum()==609
    report = []
    for label,group in [("all_40_holdouts",table),*list(table.groupby("split"))]:
        for method, rows in group.groupby("method"):
            value = rows.spearman.dropna()
            report.append(dict(endpoint="ncs_bh",split=label,method=method,n_pairs=len(rows),
                    n_spearman_defined=len(value),spearman_mean=value.mean(),spearman_sample_sd=value.std(),
                    top1_success_rate=rows.top1_success.mean(),top4_success_rate=rows.top4_success.mean(),
                    n_pairs_without_observed_success=int((~rows.any_success).sum()),
                    n_test_recipes=int(rows.n_conditions.sum()),
                    n_unseen_training_recipes=int(rows.n_train_unseen_recipes.sum())))
    chemistry = lru_cache(maxsize=None)(lambda pos: structure(raw,pos))
    chosen, inventory = select(individual,rank_examples,chemistry)
    WORK.mkdir(parents=True,exist_ok=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(report).to_csv(OUTPUT/"condition_ranking.csv",index=False)
    pd.DataFrame(chosen).to_csv(WORK/"examples_selected.csv",index=False)
    (WORK/"RANK_AND_EXAMPLES.json").write_text(json.dumps(dict(
        strict_pair_instances=609, example_inventory=inventory, min_error_gap=MIN_GAP,
        condition_ranking_sha256=digest(OUTPUT/"condition_ranking.csv"),
        examples_selected_sha256=digest(WORK/"examples_selected.csv"),
    ),indent=2,sort_keys=True)+"\n")
    print(pd.DataFrame(report).query("split=='all_40_holdouts'").to_string(index=False),flush=True)
    print(inventory,flush=True)


if __name__ == "__main__":
    main()
