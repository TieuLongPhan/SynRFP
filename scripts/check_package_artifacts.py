#!/usr/bin/env python3
"""Validate built SynRFP distributions and smoke-test the wheel."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile

#: Modules that must ship in both the wheel and the source archive.
REQUIRED_MODULES = (
    "synrfp/__init__.py",
    "synrfp/encode/core.py",
    "synrfp/graph/molecule.py",
    "synrfp/sketchers/parity_fold.py",
    "synrfp/tokenizers/wl.py",
)

#: Top-level paths that must not leak into the wheel.
FORBIDDEN_PREFIXES = ("tests/", "Experiments/", "scripts/", "doc/", "docs/")


def _one_artifact(directory: Path, pattern: str) -> Path:
    """Return the only artifact matching a glob.

    :param directory: Distribution directory.
    :type directory: Path
    :param pattern: Glob used to select an artifact.
    :type pattern: str
    :return: Matching artifact.
    :rtype: Path
    :raises RuntimeError: If the match count is not one.
    """
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {pattern!r} artifact in {directory}, found {matches}."
        )
    return matches[0]


def _check_members(members: set[str], artifact: Path, *, strict: bool) -> None:
    """Require the package modules and, for wheels, reject non-package files.

    :param members: Archive member names.
    :type members: set[str]
    :param artifact: Artifact used in error messages.
    :type artifact: Path
    :param strict: Also reject paths outside the package (wheel only).
    :type strict: bool
    :raises RuntimeError: If a required member is absent or a forbidden one is present.
    """
    missing = [
        suffix
        for suffix in REQUIRED_MODULES
        if not any(member.endswith(suffix) for member in members)
    ]
    if missing:
        raise RuntimeError(f"{artifact.name} is missing: {', '.join(missing)}")
    if strict:
        leaked = sorted(m for m in members if m.startswith(FORBIDDEN_PREFIXES))
        if leaked:
            raise RuntimeError(f"{artifact.name} ships non-package files: {leaked[:5]}")


def inspect_archives(wheel: Path, sdist: Path) -> None:
    """Inspect wheel and source archives for required files.

    :param wheel: Built wheel path.
    :type wheel: Path
    :param sdist: Built source distribution path.
    :type sdist: Path
    """
    with zipfile.ZipFile(wheel) as archive:
        _check_members(set(archive.namelist()), wheel, strict=True)
    with tarfile.open(sdist, "r:gz") as archive:
        _check_members(set(archive.getnames()), sdist, strict=False)


def smoke_test_wheel(wheel: Path, project_root: Path) -> None:
    """Install a wheel with its dependencies and fingerprint one reaction.

    :param wheel: Built wheel path.
    :type wheel: Path
    :param project_root: Repository root containing ``pyproject.toml``.
    :type project_root: Path
    """
    with (project_root / "pyproject.toml").open("rb") as handle:
        expected_version = tomllib.load(handle)["project"]["version"]

    with tempfile.TemporaryDirectory(prefix="synrfp-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", str(wheel)],
            check=True,
            cwd=root,
        )
        code = f"""
import synrfp

expected = {expected_version!r}
if synrfp.__version__ != expected:
    raise SystemExit("version mismatch: %s != %s" % (synrfp.__version__, expected))
bits = synrfp.synrfp("CCO>>C=C.O", bits=256)
if len(bits) != 256 or set(bits) - {{0, 1}}:
    raise SystemExit("unexpected fingerprint from the installed wheel")
print("synrfp %s: wheel import and fingerprint verified" % synrfp.__version__)
"""
        subprocess.run([str(python), "-c", code], check=True, cwd=root)


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    :return: Parsed command-line namespace.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("dist"),
        help="directory containing exactly one wheel and one source archive",
    )
    return parser.parse_args()


def main() -> None:
    """Run archive inspection and the isolated wheel smoke test."""
    args = parse_args()
    directory = args.directory.resolve()
    wheel = _one_artifact(directory, "*.whl")
    sdist = _one_artifact(directory, "*.tar.gz")
    project_root = Path(__file__).resolve().parents[1]
    inspect_archives(wheel, sdist)
    smoke_test_wheel(wheel, project_root)


if __name__ == "__main__":
    main()
