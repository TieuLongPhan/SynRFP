"""Configuration loading and naming conventions shared by all experiment scripts."""
from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import yaml

EXP_DIR = Path(__file__).resolve().parents[1]
log = logging.getLogger("synrfp_exp")
_RESULTS_ROOT = EXP_DIR


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load ``config.yaml`` and resolve every entry of ``paths`` to an absolute path."""
    path = Path(path) if path else EXP_DIR / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    base = path.resolve().parent
    cfg["paths"] = {k: (base / v).resolve() for k, v in cfg["paths"].items()}
    cfg["paths"]["work"].mkdir(parents=True, exist_ok=True)
    global _RESULTS_ROOT
    _RESULTS_ROOT = cfg["paths"].get("results_root", EXP_DIR)
    cfg["_config_path"] = str(path.resolve())
    return cfg


@dataclass(frozen=True)
class SynRFPConfig:
    """One point of the SynRFP design space."""

    tokenizer: str
    radius: int
    bits: int
    sketch: str
    mode: str

    @property
    def name(self) -> str:
        return (
            f"synrfp-{self.tokenizer}-r{self.radius}-b{self.bits}"
            f"-{self.sketch}-{self.mode}"
        )

    @classmethod
    def from_name(cls, name: str) -> "SynRFPConfig":
        _, tok, r, b, sk, mode = name.split("-")
        return cls(tok, int(r[1:]), int(b[1:]), sk, mode)


def default_config(cfg: Dict[str, Any]) -> SynRFPConfig:
    return SynRFPConfig(**cfg["synrfp"]["default"])


def grid_configs(cfg: Dict[str, Any]) -> List[SynRFPConfig]:
    g = cfg["synrfp"]["grid"]
    return [
        SynRFPConfig(t, r, b, s, m)
        for t, r, b, s, m in itertools.product(
            g["tokenizer"], g["radius"], g["bits"], g["sketch"], g["mode"]
        )
    ]


def resolve_fingerprint(cfg: Dict[str, Any], name: str) -> str:
    """Map aliases (``synrfp_default``) to concrete fingerprint names."""
    if name == "synrfp_default":
        return default_config(cfg).name
    return name


def expand_fingerprints(cfg: Dict[str, Any], names: List[str]) -> List[str]:
    out: List[str] = []
    for n in names:
        if n == "synrfp_grid":
            out.extend(c.name for c in grid_configs(cfg))
        else:
            out.append(resolve_fingerprint(cfg, n))
    return list(dict.fromkeys(out))


def results_dir(experiment: str) -> Path:
    """
    ``<results_root>/<experiment>/results``; each experiment keeps its own outputs.

    ``results_root`` defaults to ``Experiments/`` and can be overridden in the
    ``paths`` section of the config (used for smoke tests).
    """
    d = _RESULTS_ROOT / experiment / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def iter_json(folder: Path) -> Iterator[Dict[str, Any]]:
    for p in sorted(folder.rglob("*.json")):
        yield json.loads(p.read_text())
