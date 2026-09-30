# Workspace cleanup

The generated experiment outputs were cleaned on 2026-10-01 after the
benchmark and factorial runs completed.

Kept:

- raw inputs and dataset metadata under `Experiments/data/` and
  `Experiments/01_data/results/`;
- per-cell classification results and compact prediction artifacts needed to
  reproduce the reported tables;
- clustering, robustness, reaction-center, enzyme-activity, interpretation,
  ANOVA, and report outputs;
- analysis notes, run summaries, and protocol/audit reports.

Removed:

- the rebuildable `Experiments/work/` cache (feature matrices, tables,
  canonical arrays, and downloaded intermediate data; about 27.5 GB);
- stale queue locks, PID files, process-ID lists, Python bytecode caches, and
  `.pytest_cache/`.

No standalone checkpoint files were present in the repository. MLP pilot split
archives did contain initialized layer coefficients alongside split metadata;
those archives and their large local feature inputs remain on disk and are
excluded from version control. The `Experiments/.gitignore` rules explicitly
exclude these local pilot artifacts, common checkpoint formats, and
`weights/` or `checkpoints/` directories so trained or initialized model
weights stay local.
