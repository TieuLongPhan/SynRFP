"""Audit the Schneider and USPTO-50K taxonomy joins against pinned releases."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import file_digest
import io
import json
from pathlib import Path
import pickletools
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase

HERE = Path(__file__).resolve().parent
BASE = Path.home() / ".local/share/synrfp-runs/request81_class_names_v1"
OUT = HERE / "publication_metadata/class_names_v1"
SEP = re.compile(r"(?<!-)>")
MAP = re.compile(r":(\d+)\]")
RDLogger.DisableLog("rdApp.*")
SYN = "e7c03766bd6a405da468e281d922cf1beccdd56a"
CAT = "5f34a26cb90c95831268c635de3dea1be3b7552b"
RXNFP = "6fd48f4927c2178555cc5d71dbfb225fb178f43c"
LIU = "9a6f040198ce990cbdeea812e7e99df7becd142c"
NAMES = "ChemReactionClassification/data/names_rTypes_classes_superclasses_training_test_set_patent_data.pkl"
RAW = {
    "schneider_b": ("schneider50k_balanced", "reactions", "y", "sch"),
    "schneider_u": ("schneider50k_unbalanced", "rxn", "y", "sch"),
    "uspto_50k_b": ("USPTO_50k_balanced", "reactions", "new_class", "USPTO"),
    "uspto_50k_u": ("USPTO_50k_unbalanced", "reactions", "new_class", "USPTO"),
}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def safe_dictionary(data):
    # The two historical author dictionaries use only string literals and basic
    # dict-building opcodes. Parse their literal pairs without unpickling code.
    ops = list(pickletools.genops(data))
    assert {op.name for op, _, _ in ops} <= {"MARK", "DICT", "PUT", "STRING", "SETITEM", "STOP"}
    values = [value for op, value, _ in ops if op.name == "STRING"]
    assert len(values) % 2 == 0 and len(set(values[::2])) == len(values)//2
    return dict(zip(values[::2], values[1::2], strict=True))


def align(root):
    manifest = json.loads((root/"synrxn_manifest.json").read_text())
    encoded = json.dumps(manifest)
    panels, stats = {}, {}
    for s, (parent_name, react, label, prefix) in RAW.items():
        s_path = root/f"{s}.csv.gz"
        manifest_key = f"classification/{s}.csv.gz"
        match = re.search(rf'"key": "{re.escape(manifest_key)}".*?"sha256": "([a-f0-9]{{64}})"', encoded)
        assert match and digest(s_path) == match.group(1)
        child = pd.read_csv(s_path, keep_default_na=False)
        parent = pd.read_csv(root/f"cat_{parent_name}.csv.gz", keep_default_na=False)
        np.testing.assert_array_equal(child.r_id, [f"{prefix}_{i+1}" for i in range(len(parent))])
        for src, dst in ((react, "rxn"), (label, "label"), ("split", "split")):
            np.testing.assert_array_equal(child[dst], parent[src])
        panels[s] = child
        stats[s] = dict(exact_smiles_and_row_key=len(child), identical_labels=len(child),
                        identical_splits=len(child), synrxn_path=manifest_key,
                        syncat_path=f"Data/raw/{parent_name}.csv.gz")
    return panels, stats


@lru_cache(maxsize=16000)
def canonical(smi):
    text = MAP.sub("]", smi)
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol), mol.GetNumHeavyAtoms(), sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())


def key(left, right):
    inputs, products = [canonical(m) for m in left], [canonical(m) for m in right]
    if not inputs or not products or any(m is None for m in inputs + products):
        return None
    return tuple(sorted(m[0] for m in inputs)), tuple(sorted(m[0] for m in products))


def actual(text):
    left, middle, right = SEP.split(text)
    lhs = ".".join(s for s in (left, middle) if s)
    return key(lhs.split("."), right.split("."))


def original(record):
    cls, namerxn, indigo = record
    keys = set()
    for text in (namerxn, indigo):
        if text.count(">") < 2:
            continue
        left, middle, right = SEP.split(text)
        precursors = left.split(".")
        products = right.split(".")
        ranked = [(canonical(smiles), smiles) for smiles in products]
        ranked = [(score, smiles) for score, smiles in ranked if score is not None]
        if not ranked:
            continue
        best_size = max(score[1:] for score, _ in ranked)
        for score, product in ranked:
            if score[1:] != best_size:
                continue
            product_maps = set(MAP.findall(product))
            contributing = [p for p in precursors if product_maps.intersection(MAP.findall(p))]
            all_left = precursors + ([*middle.split(".")] if middle else [])
            for cohort in (contributing, precursors, all_left):
                result = key(cohort, [product])
                if result:
                    keys.add(result)
    return int(cls), keys


def liu(record):
    source, target = record
    tokens = source.split()
    found = re.fullmatch(r"<RX_(\d+)>", tokens[0])
    assert found is not None
    result = key("".join(target.split()).split("."), "".join(tokens[1:]).split("."))
    return int(found.group(1)), {result} if result else set()


def lookup(pairs):
    target_classes = defaultdict(set)
    no_key = 0
    for cls, keys in pairs:
        no_key += not keys
        for serialized in keys:
            target_classes[serialized].add(cls)
    return target_classes, no_key


def votes(index, keys, labels):
    counts = Counter({key: 0 for key in ("agree_unique", "disagree_unique", "ambiguous_classes", "unmatched",
                                          "synrxn_parse_refusals")})
    confusion = Counter()
    for serialized, lab in zip(keys, labels, strict=True):
        matches = index.get(serialized, set())
        counts["synrxn_parse_refusals"] += serialized is None
        if not matches:
            counts["unmatched"] += 1
        elif len(matches) > 1:
            counts["ambiguous_classes"] += 1
        else:
            candidate, = matches
            confusion[lab, candidate] += 1
            counts["agree_unique" if lab+1 == candidate else "disagree_unique"] += 1
    return dict(counts), confusion


def emit(root, panels, stats, n_jobs):
    code2id = json.loads((root/"rxnclass2id.json").read_text())
    names = json.loads((root/"rxnclass2name.json").read_text())
    with ZipFile(root/"ci5006614_si_002.zip") as source:
        names_original = safe_dictionary(source.read(NAMES))
    assert set(code2id.values()) == set(range(50))
    for code in code2id:
        assert names[code] == names_original[code]
    original_schneider = pd.read_csv(root/"schneider50k.tsv", sep="\t", keep_default_na=False)
    np.testing.assert_array_equal(original_schneider.rxn, panels["schneider_u"].rxn)
    np.testing.assert_array_equal(original_schneider.rxn_class.map(code2id), panels["schneider_u"].label)
    for stem in ("schneider", "uspto_50k"):
        a, b = (panels[f"{stem}_{suffix}"] for suffix in ("u", "b"))
        np.testing.assert_array_equal(a.r_id, b.r_id)
        np.testing.assert_array_equal(a.label, b.label)
        np.testing.assert_array_equal(a.split, b.split)
    source_name = "https://doi.org/10.1021/ci5006614.s002"
    name_table = [dict(index=index, code=code, name=names[code], source=source_name) for code, index in code2id.items()]
    pd.DataFrame(name_table).sort_values("index").to_csv(OUT/"schneider_class_names.csv", index=False)

    with ZipFile(root/"ci6b00564_si_002.zip") as source:
        class_names = safe_dictionary(source.read("data/rxnClasses.pickle"))
        with source.open("data/dataSetB.csv") as stream:
            raw = pd.read_csv(io.TextIOWrapper(stream), usecols=[
                "rxn_Class", "rxnSmiles_Mapping_NameRxn", "rxnSmiles_Mapping_IndigoTK"], keep_default_na=False)
    xml = ET.parse(root/"liu.xml")
    liu_classes = {}
    for row in xml.findall('.//table-wrap[@id="tbl1"]//tbody/tr'):
        cells = [" ".join("".join(c.itertext()).split()) for c in row]
        liu_classes[int(cells[0])] = cells[1]
    assert set(liu_classes) == {int(n) for n in class_names} == set(range(1,11))
    assert all(liu_classes[k].lower().replace("–", "-") == class_names[str(k)].lower() for k in liu_classes)
    liu_rows = []
    for split in ("train", "valid", "test"):
        lhs = (root/f"{split}_sources").read_text().splitlines()
        rhs = (root/f"{split}_targets").read_text().splitlines()
        liu_rows.extend(zip(lhs, rhs, strict=True))
    assert len(liu_rows) == 50037 and len(raw) == 50000
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        index_source, source_refusals = lookup(pool.map(original, raw.itertuples(index=False, name=None), chunksize=64))
        print("Original USPTO lookup complete", len(index_source), flush=True)
        index_liu, liu_refusals = lookup(pool.map(liu, liu_rows, chunksize=64))
        print("Liu USPTO lookup complete", len(index_liu), flush=True)
        keys = list(pool.map(actual, panels["uspto_50k_u"].rxn, chunksize=64))
    labels = panels["uspto_50k_u"].label
    comparisons = []
    for reference, index, size, no_key in (
        ("Schneider 2016 Data Set B", index_source, len(raw), source_refusals),
        ("Liu 2017 published model input", index_liu, len(liu_rows), liu_refusals),
    ):
        counts, mat = votes(index, keys, labels)
        comparisons.append(dict(reference=reference, source_rows=size, no_source_key=no_key,
                                **counts, confusion=[dict(synrxn=int(a), author=b, n=n)
                                                    for (a,b),n in sorted(mat.items())]))
    counts, mat = votes({k: index_source.get(k, set()) | index_liu.get(k, set()) for k in keys}, keys, labels)
    comparisons.append(dict(reference="Either release, ambiguous if releases differ", source_rows=None,
                            no_source_key=None, **counts, confusion=[dict(synrxn=int(a), author=b, n=n)
                                                                      for (a,b),n in sorted(mat.items())]))
    rows = []
    for lab in range(10):
        rec = dict(synrxn_label=lab, author_class=lab+1, name=liu_classes[lab+1])
        for short, check in zip(("schneider2016", "liu2017", "combined"), comparisons, strict=True):
            rec[short+"_n_matching_class"] = sum(r["n"] for r in check["confusion"]
                                               if r["synrxn"] == lab and r["author"] == lab+1)
            rec[short+"_n_conflicting_class"] = sum(r["n"] for r in check["confusion"]
                                                  if r["synrxn"] == lab and r["author"] != lab+1)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT/"uspto50k_label_crosswalk.csv", index=False)
    return stats, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=BASE)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    root = args.inputs
    OUT.mkdir(parents=True, exist_ok=True)
    pins = json.loads((HERE/"inputs.json").read_text())
    for name, spec in pins.items():
        assert digest(root/name) == spec["sha256"]
    stats, comparisons = emit(root, *align(root), args.jobs)
    (OUT/"AUDIT.json").write_text(json.dumps(dict(
        status="complete", rxnfp=RXNFP, syncat=CAT, synrxn=SYN, liu=LIU, rdkit=rdBase.rdkitVersion,
        pandas=pd.__version__, numpy=np.__version__, inputs=pins, synrxn_exact_lineage=stats,
        comparisons=comparisons), indent=2)+"\n")
    print(json.dumps([{k:v for k,v in r.items() if k!="confusion"} for r in comparisons],indent=2), flush=True)


if __name__ == "__main__":
    main()
