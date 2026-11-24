# EnzymeMap interval reproduction

The archive contains 10,000 paired scaffold-group bootstrap draws for the
eight comparison estimates in [scores.json](../scores.json). In each draw,
both targets and all arms use the same sampled scaffold groups. Read the
`contrasts` array for column labels. The primary interaction is the EC level-2
gain minus the bond-edit gain, both versus standard delta.

From this directory, NumPy can recover every published percentile interval:

```python
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

receipt = json.loads(Path("receipt.json").read_text())
path = Path("paired_bootstrap_draws.npz")
assert sha256(path.read_bytes()).hexdigest() == receipt["published_draws_sha256"]

with np.load(path, allow_pickle=False) as saved:
    keys = [tuple(row) for row in saved["contrasts"]]
    draws = saved["draws"]

scores = json.loads(Path("../scores.json").read_text())
tails = [0.025, 0.975]
for row in scores["estimates"]:
    column = keys.index((row["target"], row["comparator"]))
    np.testing.assert_allclose(np.quantile(draws[:, column], tails),
                               row["interval"], atol=1e-15, rtol=0)

ec = draws[:, keys.index(("ec_level_2", "d2048"))]
bond = draws[:, keys.index(("bond_edit_multiset", "d2048"))]
np.testing.assert_allclose(np.quantile(ec - bond, tails),
                           scores["interaction"]["interval"], atol=1e-15, rtol=0)
```

The unrounded lower limits are 0.447746 percentage points for the EC level-2
gain and 0.445195 points for the interaction. Rounding to two decimal places
gives +0.45 for both. The accompanying receipt records the original workstation
archive checksum and the error rates reconstructed from all three fitting seeds.
The export contains the aggregate draws and their column labels.

## Submission checks

| Target | Standard-delta group-weighted error | Pooled-context group-weighted error |
| --- | ---: | ---: |
| EC level 2 | 6.365% | 4.256% |
| Bond edits | 15.791% | 16.396% |

All four pooled-versus-reactant-only intervals span zero across the two
EnzymeMap targets and the two [QuantumPioneer targets](../../quantumpioneer_v1/scores.json).
The archived scores and bootstrap draws reproduce those intervals.

One scaffold group contains 8,060 of the 9,631 admitted EnzymeMap test rows.
The primary analysis gives each of the 420 scaffold groups equal weight.
As a descriptive influence check, omitting that group from the held-out
group gains gives an interaction of +2.710 percentage points across 419
remaining groups (95% paired interval +0.445 to +5.085). This check uses the
existing fitted models. The primary interaction remains +2.714 percentage
points (95% paired interval +0.445 to +5.098).
