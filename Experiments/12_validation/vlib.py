"""Shared helpers of the Validation suite (Experiments/12_validation).

The suite never writes into the numbered experiment folders or ``work/``.
Each input policy has its own config (``configs/<policy>.yaml``), cache
(``../work/validation/<policy>``), and result root (``runs/<policy>``), so the
existing scripts run unchanged with ``--config 12_validation/configs/<policy>.yaml``.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
# VALIDATION_SMOKE=1 redirects every script to the smoke/ tree written by
# `00_setup.py --smoke N`.
ROOT = HERE / "smoke" if os.environ.get("VALIDATION_SMOKE") else HERE
RUNS = ROOT / "runs"
CONFIGS = ROOT / "configs"
for p in (str(EXP), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
# joblib/loky workers do not inherit sys.path
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(HERE), str(EXP), str(REPO)]
    + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])


def study_settings() -> Dict[str, Any]:
    return yaml.safe_load((HERE / "study.yaml").read_text())


def policy_config_path(policy: str) -> Path:
    return CONFIGS / f"{policy}.yaml"


def load_policy(policy: str) -> Dict[str, Any]:
    """Experiment config of one input policy (written by 00_setup.py)."""
    from synrfp_exp.config import load_config

    path = policy_config_path(policy)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run 00_setup.py first")
    return load_config(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _run(cmd: List[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:  # tool not installed on this host
        return f"<unavailable: {e}>"


def environment() -> Dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    pk = {}
    for name in ("rdkit", "drfp", "numpy", "scipy", "pandas", "scikit-learn", "joblib",
                 "synkit", "synrxn", "PyYAML", "bblean", "statsmodels"):
        try:
            pk[name] = version(name)
        except PackageNotFoundError:
            pk[name] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "packages": pk,
        "synrfp_commit": _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "synrfp_package_dirty": bool(_run(["git", "-C", str(REPO), "status", "--porcelain",
                                           "--", "synrfp", "Experiments/synrfp_exp"])),
    }


def write_manifest(name: str, outputs: List[Path], extra: Dict[str, Any] | None = None) -> None:
    doc = {"step": name, "environment": environment(),
           "study_yaml_sha256": sha256(HERE / "study.yaml"),
           "script_sha256": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
           "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in outputs if p.exists()}}
    doc.update(extra or {})
    out = RUNS / "manifests" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=str))


# ---------------------------------------------------------------------------
# Input policies
# ---------------------------------------------------------------------------
def _canonical_fields(rsmi: str) -> List[List[str]] | None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    parts = rsmi.strip().split(" ")[0].split(">")
    if len(parts) != 3:
        return None
    out = []
    for field in parts:
        smis = []
        for s in (x for x in field.split(".") if x):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return None
            for a in mol.GetAtoms():
                a.SetAtomMapNum(0)
            smis.append(Chem.MolToSmiles(mol))
        out.append(sorted(smis))
    return out


def apply_policy(rsmi: str, policy: str) -> Tuple[str, str]:
    """Return ``(string, status)``.

    ``status`` is ``ok``, ``unchanged`` (raw policy), or ``fallback_raw`` when
    a component cannot be parsed by RDKit. A fallback keeps the raw string so
    that the policy never creates a failure the raw input did not have; these
    rows are counted in the setup report.
    """
    if policy == "raw":
        return rsmi, "unchanged"
    f = _canonical_fields(rsmi)
    if f is None:
        return rsmi, "fallback_raw"
    reac, agent, prod = f
    if policy == "n2":
        left, right = Counter(reac + agent), Counter(prod)
        common = left & right
        if common and (left - common) and (right - common):
            left, right = left - common, right - common
            return (".".join(sorted(left.elements())) + ">>"
                    + ".".join(sorted(right.elements()))), "ok"
    elif policy != "n1":
        raise ValueError(f"unknown policy {policy!r}")
    return ">".join(".".join(x) for x in (reac, agent, prod)), "ok"


def randomized_rsmi(rsmi: str, seed: int) -> str:
    """Chemically equivalent re-serialization: random atom order per component
    and shuffled component order within each field. Empty if unparsable."""
    import numpy as np
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    rng = np.random.default_rng(seed)
    out = []
    for field in rsmi.strip().split(" ")[0].split(">"):
        smis = []
        for s in (x for x in field.split(".") if x):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return ""
            order = rng.permutation(mol.GetNumAtoms()).tolist()
            smis.append(Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False))
        rng.shuffle(smis)
        out.append(".".join(smis))
    return ">".join(out)


# Equivalent spellings must map to one n1 string; distinct chemistry must not.
POLICY_CASES = {
    "equal": [
        ("[CH3][OH].[CH3][C](=[O])[OH]>>[CH3][C](=[O])[O][CH3].[OH2]",
         "CO.CC(=O)O>>CC(=O)OC.O"),                        # bracketed hydrogens
        ("C1=CC=CC=C1Br.OB(O)c1ccccc1>>c1ccc(-c2ccccc2)cc1",
         "OB(O)c1ccccc1.Brc1ccccc1>>c1ccc(cc1)-c1ccccc1"),  # Kekule form, order
        ("[CH3:1][OH:2].[Na+:3].[Cl-:4]>>[CH3:1][Cl:4].[Na+:3].[OH-:2]",
         "[Cl-].[Na+].CO>>CCl.[OH-].[Na+]"),               # maps, ions, order
        ("OCC>[Pd]>CC=O", "CCO>[Pd]>O=CC"),                 # agent field kept
    ],
    "distinct": [
        ("C[C@H](N)C(=O)O>>C[C@H](N)C(=O)OC", "C[C@@H](N)C(=O)O>>C[C@@H](N)C(=O)OC"),  # stereo
        ("CC(=O)C>>CC(O)C", "CC(O)=C>>CC(O)C"),             # tautomers are not merged
        ("[13CH3]O>>[13CH3]Cl", "CO>>CCl"),                 # isotopes kept
        ("CC(=O)[O-].[Na+]>>CC(=O)OC", "CC(=O)O>>CC(=O)OC"),  # salts/charges kept
        ("CCO>>CC=O", "CC=O>>CCO"),                         # direction kept
    ],
}


def policy_selftest() -> None:
    for a, b in POLICY_CASES["equal"]:
        ca, cb = apply_policy(a, "n1"), apply_policy(b, "n1")
        assert ca[1] == cb[1] == "ok" and ca[0] == cb[0], (a, b, ca, cb)
    for a, b in POLICY_CASES["distinct"]:
        assert apply_policy(a, "n1")[0] != apply_policy(b, "n1")[0], (a, b)
    s, st = apply_policy("CCO.O.[Pd]>>CC=O.O.[Pd]", "n2")
    assert (s, st) == ("CCO>>CC=O", "ok"), s
    assert apply_policy("CCO.O>>CCO.O", "n2")[0] == "CCO.O>>CCO.O"  # nothing would remain
    assert apply_policy("not a reaction", "n1")[1] == "fallback_raw"
    for a, _ in POLICY_CASES["equal"] + POLICY_CASES["distinct"]:
        r = randomized_rsmi(a, 3)
        assert apply_policy(r, "n1")[0] == apply_policy(a, "n1")[0], a
