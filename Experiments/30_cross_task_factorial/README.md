# Complete seven-task construction analysis

This is a new sealed extension of the completed 72-construction classification,
clustering and yield factorial. It completes the same grid for barrier,
feasibility, reagent and condition prediction while preserving each native
dataset, split, population and MLP. See [PROTOCOL.md](PROTOCOL.md).

Full scope: 6,480 logical cells, 270 verified native reuse candidates and 6,210 fresh
fits. Primary-only barrier scope: 3,960 logical cells and 3,690 fresh fits. The grid
crosses 3 tokenizers × 2 radii × 3 sketches × 4 modes at 2,048 total bits and seed 1. Original
roles is retained; the retired joint-role sketch is excluded.

Local run directory:
`~/.local/share/synrfp-runs/cross_task_factorial_v1`. Weights, shared label
payloads, packed fingerprints and compact exact predictions remain local. Only
reviewed protocols, coverage, audit receipts and result summaries are intended
for version control. No global preset is selected automatically.

The root scheduler executes these stages in capped services:

```bash
python Experiments/30_cross_task_factorial/study.py --preflight
python Experiments/30_cross_task_factorial/study.py --prepare \
  --barrier-scope all_activation \
  --proof /absolute/path/to/encoder-proof.json \
  --proof /absolute/path/to/native-worker-proof.json
python Experiments/30_cross_task_factorial/encoder.py --workers 8
```

Use the existing `synrfp` environment for selector/encoder work and `synprop`
for model work. The preparation command refuses missing proof receipts and
refuses a changed scope after sealing. It writes only this version's derived
selectors and receipts. Historical data, reports, source files and model weights
are immutable inputs. Production workers start only through the root resource
controller with the original exclusive physical-device leases.
