"""Shared manifest and source helpers for Request 86."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = REPO.parent / "SynRFP"
BASE = Path.home() / ".local/share/synrfp-runs"
ROOT = BASE / "request86_invariance_v1"
TMP = Path("/var/tmp/synrfp-labhhc4-storage")
P22 = BASE / "all_endpoint_factorial_full_v1"
P52 = BASE / "coverage_refits_v1"
OUTPUT = HERE / "publication_metadata/invariance_v1"
ENDPOINTS = ("schneider_ub","tpl_b","ecreact_1","mech_2")
FORMS = ("original","randomised","reversed")
D = "synrfp-wl-r2-b2048-parity-delta"
DU = "synrfp-wl-r2-b2048-parity-delta_union"
BASELINES = ("drfp","rdkit_diff","rdkit_struct","crxnfp","rxnfp_mlm")
FPS = (D,DU,*BASELINES)
CANON = "drfp_on_canonicalised_input"
BINARY = {D,DU,"drfp","rdkit_struct",CANON}
SEP = re.compile(r"(?<!-)>")
OUT86 = "6e7acef5d5f16d9b25ed6009a7538bd1fe638482"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,data):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
    tmp.replace(path)


def bound(path,expected=None):
    value=digest(path)
    if expected is not None and value!=expected:
        raise ValueError(f"Changed frozen input {path}")
    return value


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m


def failure(error,location):
    # The detailed record is retained even where a native parser omits the
    # underlying diagnostic. Classify stage and cause independently.
    msg=str(error).strip().splitlines()
    msg=" | ".join(msg)[:600]
    text=(type(error).__name__+" "+msg).lower()
    if any(s in text for s in ("valence","atomvalence","kekuliz","aromaticity")):
        cause="invalid_valence_or_aromaticity"
    elif any(s in text for s in ("length","512","position embedding")) and location in ("tokenizer","model"):
        cause="token_limit_or_shape"
    elif any(s in text for s in ("separator","too many values to unpack","three fields","reaction smiles")):
        cause="reaction_separator_or_format"
    elif any(s in text for s in ("mol","smiles","parse","sanitiz","wrong mol")):
        cause="molecule_parse_or_sanitization"
    else:
        cause=location+"_refusal"
    return dict(stage=location,cause=cause,exception=type(error).__name__,message=msg)
