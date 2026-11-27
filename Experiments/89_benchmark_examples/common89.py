"""Shared pinned readers and structure-selection helpers for Request 89."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
ORIGINAL = WORKSPACE/"SynRFP"
REQ88 = WORKSPACE/"SynRFP-witness-examples/Experiments/88_witness_examples"
REQ78 = WORKSPACE/"SynRFP-clustering-barrier-ceiling/Experiments/78_clustering_barrier_ceiling"
sys.path[:0] = [str(REQ88), str(REQ78)]

from common88 import BASE, chemistry
from common78 import Pins, digest, export, read, write
from select88 import canonical_sides

OUTPUT = HERE/"publication_metadata/benchmark_examples_v1"
TMP = Path("/var/tmp/synrfp-labhhc4-storage/runs")


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m


def feature_row(endpoint,pair,identifier,rxn,position,label):
    size, scaffold = chemistry(rxn)
    return dict(endpoint=endpoint,pair_id=pair,side=1,r_id=str(identifier),rxn=rxn,label=label,
                n_heavy_atoms=size,scaffold=scaffold,source_position=int(position),
                mapped_rxn="",mapping_origin="")
