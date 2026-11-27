"""Run the Request 86 audit on TPL unbalanced alone."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "86_invariance_all_fps"
sys.path.insert(0, str(SOURCE))

REQUEST87 = "6259067a522ee0627b705ae7c49fb48aa197a192"
REQUEST86 = "3fdd44463fe4cdec76a92e62ecc6245474c650f6"


def note(output, failures, warnings):
    if len(failures):
        counts = sum(int(v) for v in failures["count"])
        rows = failures["example_row_id"].nunique()
        status = f"The failure ledger contains {counts} encoder/form refusals. "
        status += f"It shows {rows} distinct example row IDs and these causes: "
        status += ", ".join(sorted(failures["cause"].unique())) + "."
    else:
        status = (
            "All 2,000 sampled rows encoded in original, rewritten and reversed form "
            "for every fingerprint and the canonical DRFP control. No sampled encoding "
            "refusals were observed."
        )
    count = sum(int(v) for v in warnings["count"])
    message = (
        "# TPL unbalanced invariance audit\n\n"
        "The 2,000-row sample uses the ordered TPL unbalanced release panel and seed 0. "
        "The seven benchmark arms and canonical-input DRFP control reuse the Request 86 encoders. "
        f"{status} Successful encodings produced {count} diagnostics across the three forms. "
        "These sample counts concern this audit's rows. The other three endpoints retain their "
        "Request 86 results.\n"
    )
    (output / "note.md").write_text(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "classical", "crxnfp", "rxnfp_mlm", "report"))
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()

    import common86 as common

    common.ROOT = common.BASE / "request87_invariance_tpl_ub_v1"
    common.ENDPOINTS = ("tpl_ub",)
    common.OUT86 = REQUEST87
    common.OUTPUT = HERE / "publication_metadata/invariance_tpl_ub_v1"
    if args.phase == "prepare":
        import prepare86
        prepare86.prepare()
        return
    if args.phase == "classical":
        import classical86
        sys.argv = [sys.argv[0], "--jobs", str(args.jobs)]
        classical86.main()
        return
    if args.phase in ("crxnfp", "rxnfp_mlm"):
        import neural86
        sys.argv = [sys.argv[0], args.phase, "--jobs", str(args.jobs)]
        neural86.main()
        return

    import pandas as pd
    import report86

    # Keep the original reporting arithmetic and use request-scoped filenames.
    report86.artifact_names.update(
        scores="perturbation_tpl_ub.csv",
        failures="encoding_failures_tpl_ub.csv",
        warnings="encoding_warnings_tpl_ub.csv",
        ledger="encoding_failure_rows_tpl_ub.csv",
    )
    report86.markdown = lambda scores, issues: note(
        common.OUTPUT, issues[issues.kind.eq("failure")], issues[issues.kind.eq("warning")]
    )
    report86.main()
    path = common.OUTPUT / "COMPLETE.json"
    manifest = common.read(path)
    family = [pd.read_csv(common.OUTPUT / filename) for filename in report86.artifact_names.values()]
    archive86 = SOURCE / "publication_metadata/invariance_v1"
    for table, original in zip(
        family,
        ("perturbation_all_fps.csv", "encoding_failures.csv", "encoding_warnings.csv",
         "encoding_failure_rows.csv"),
    ):
        assert list(table.columns) == list(pd.read_csv(archive86 / original, nrows=0).columns)
    scores, _, _, _ = family
    assert set(scores.dataset) == {"tpl_ub"} and len(scores) == 8 and (scores.n == 2000).all()
    assert int(scores.benchmark_arm.sum()) == 7
    manifest.update(request86_reference_commit=REQUEST86,
                    request87_runner_sha256=common.digest(Path(__file__)))
    common.write(path, manifest)


if __name__ == "__main__":
    main()
