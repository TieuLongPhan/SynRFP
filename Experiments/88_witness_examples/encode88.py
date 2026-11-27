"""Replay sign-only support using one representative of every exact D class."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import argparse

import numpy as np

from common88 import HERE, ROOT, InputAudit, counts, decode, frozen_source, init_native, sign, write

PANELS = ("ecreact_1", "tpl_b", "uspto50k_ub")


def process(batch):
    output = np.empty(len(batch), dtype="V32")
    for i, (text, repaired, expected) in enumerate(batch):
        delta, *_ = decode(text, repaired)
        assert counts.signed_digest(delta) == expected
        output[i] = sign(delta)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    audit = InputAudit()
    repair = init_native(audit)
    for endpoint in PANELS:
        data = frozen_source(audit, endpoint)
        values, idx, inv = np.unique(data["keys"], return_index=True, return_inverse=True)
        samples = [(data["texts"][i], data["source"] == "tpl_b" and int(data["pos"][i]) in repair, bytes(k))
                   for i, k in zip(idx, values, strict=True)]
        print(endpoint, len(samples), "exact D representatives", flush=True)
        chunks = [samples[i:i+256] for i in range(0, len(samples), 256)]
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("fork")) as executor:
            keys = np.concatenate(list(executor.map(process, chunks)))[inv]
        art = ROOT / f"{endpoint}_sign.npz"
        np.savez_compressed(art, keys=keys, source_positions=data["pos"])
        write(ROOT/f"{endpoint}_sign.json", dict(sha256=audit.use(art).name and audit.pins[str(art)],
              source_sha256=audit.pins, representative_count=len(values),
              profile="WL radius 2, exact signed counts reduced to ±1", code_sha256={
                  name: (audit.use(HERE/name) and audit.pins[str(HERE/name)])
                  for name in ("common88.py", "encode88.py")}))
        print(endpoint, len(np.unique(keys)), "sign keys", flush=True)


if __name__ == "__main__":
    main()
