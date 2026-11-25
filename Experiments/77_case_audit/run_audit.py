"""Run Request 77 from frozen archives and export aggregate audit tables."""
import argparse
import importlib.metadata
from pathlib import Path
import platform

from audit_io import Archive, HERE, OUTPUT, digest, write
from external_audit import external_reports
from path_overlap import path_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT, type=Path)
    args = parser.parse_args()
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty result directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    archive = Archive()
    enzyme, qp = external_reports(archive, output)
    paths = path_reports(archive, output)
    stats = {
        "commits": archive.commits,
        "input_sha256": archive.pins,
        "code_sha256": {p.name: digest(p) for p in sorted(HERE.glob("*.py"))},
        "packages": {package: importlib.metadata.version(package)
                     for package in ("numpy", "pandas", "rdkit", "scipy")},
        "python": platform.python_version(),
        "new_fits": 0,
        "eligibility_counts": {"enzymemap": enzyme["totals"], "quantumpioneer": qp["totals"]},
        "held_out_rows": {"enzymemap": enzyme["rows"], "quantumpioneer": qp["rows"]},
        "verified_quantumpioneer_abstractions": qp["verified_abstractions"],
        "enzymemap_carbohydrate_annotation": enzyme["carbohydrate_annotations"],
        "enzymemap_selected_loss_carbohydrate_annotation_count": enzyme["selected_carbohydrate_annotations"],
        "quantumpioneer_selected_examples_barrier_abs_error_median": {
            f"{direction}_{arm}": value for (direction, arm), value in qp["examples_barrier"].items()
        },
        "path_endpoints": len(paths),
    }
    write(output / "AUDIT.json", stats)


if __name__ == "__main__":
    main()
