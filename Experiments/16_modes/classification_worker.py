#!/usr/bin/env python
"""Persistent CPU implementation of the frozen classification MLP arithmetic.

Prepared initialization, labels and random-state arrays use the original
mlp_gpu_queue.prepare_arrays contract. Test labels are never supplied here.
The Adam update, accuracy stopping rule and restored best weights follow
12_validation/crxnfp_full_worker.py; CPU arithmetic is a separate backend.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


def atomic_npy(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def atomic_npz(path, **values):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def load_matrix(directory, stem):
    """Load retained, losslessly compressed inputs or an uncompressed fixture."""
    compressed = directory / (stem + ".npz")
    if compressed.exists():
        with np.load(compressed, allow_pickle=False) as archive:
            if set(archive.files) != {"X"}:
                raise ValueError("Feature input must contain X only, never test outcomes")
            return archive["X"]
    return np.load(directory / (stem + ".npy"), mmap_mode="r", allow_pickle=False)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(meta, arrays, features):
    required = {"train", "val", "y", "w0", "w1", "b0", "b1",
                "rng_keys", "rng_pos", "rng_has_gauss", "rng_gauss"}
    if set(arrays) != required:
        raise ValueError("Unexpected prepared state fields")
    if features.ndim != 2 or features.dtype != np.float32:
        raise ValueError("Prepared training features must be a float32 matrix")
    if len(arrays["y"]) != len(features):
        raise ValueError("Prepared labels and training features differ in length")
    train, validation = arrays["train"], arrays["val"]
    if len(train) == 0 or len(validation) == 0:
        raise ValueError("Training and internal validation must both be nonempty")
    if not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(validation.dtype, np.integer):
        raise ValueError("Prepared partitions must contain integer positions")
    if not np.array_equal(np.sort(np.r_[train, validation]), np.arange(len(features))):
        raise ValueError("Prepared training/validation partitions are not disjoint and exhaustive")
    p = meta["params"]
    if p["hidden"] != [512] or p["alpha"] != 0.0001 or p["batch_size"] != 256:
        raise ValueError("Classification recipe differs from the frozen comparison")
    if p["max_iter"] != 200:
        raise ValueError("Classification epoch cap differs from the frozen comparison")
    if arrays["w0"].shape != (features.shape[1], 512):
        raise ValueError("Input-layer shape mismatch")
    outputs = arrays["w1"].shape[1]
    if arrays["w1"].shape != (512, outputs) or arrays["b0"].shape != (512,) or arrays["b1"].shape != (outputs,):
        raise ValueError("Output-layer shape mismatch")
    y = arrays["y"]
    if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0):
        raise ValueError("Prepared labels must be nonnegative encoded integers")
    if y.max() >= (2 if outputs == 1 else outputs):
        raise ValueError("Prepared labels exceed the output catalogue")
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite prepared training features")


def train(directory, device="cpu"):
    if device != "cpu":
        raise ValueError("This separately registered worker supports CPU only")
    directory = Path(directory)
    meta = json.loads((directory / "job.json").read_text())
    for name, expected in meta.get("inputs_sha256", {}).items():
        if Path(name).name != name or file_sha(directory / name) != expected:
            raise ValueError("Prepared input receipt mismatch: " + name)
    p = meta["params"]
    with np.load(directory / "state.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    start = time.perf_counter()
    X = load_matrix(directory, "train")
    validate_inputs(meta, arrays, X)
    xt = torch.tensor(X[arrays["train"]], device=device)
    xv = torch.tensor(X[arrays["val"]], device=device)
    yt = torch.tensor(arrays["y"][arrays["train"]], device=device)
    yv = torch.tensor(arrays["y"][arrays["val"]], device=device)
    names = ["w0", "w1", "b0", "b1"]
    weights = [torch.tensor(arrays[key], device=device) for key in names]
    binary = weights[1].shape[1] == 1
    moments = [torch.zeros_like(value) for value in weights]
    variances = [torch.zeros_like(value) for value in weights]
    rng = np.random.RandomState()
    rng.set_state(("MT19937", arrays["rng_keys"], int(arrays["rng_pos"]),
                   int(arrays["rng_has_gauss"]), float(arrays["rng_gauss"])))
    order = np.arange(len(xt))
    best_score, stale, step = -np.inf, 0, 0
    history = []
    batch_size = min(p["batch_size"], len(xt))

    def predict_tensor(data, values):
        w0, w1, b0, b1 = values
        for offset in range(0, len(data), 8192):
            logits = (data[offset:offset + 8192] @ w0 + b0).relu() @ w1 + b1
            yield (logits[:, 0] > 0).long() if binary else logits.argmax(1)

    with torch.no_grad():
        for epoch in range(p["max_iter"]):
            order = rng.permutation(order)
            indices = torch.tensor(order, device=device)
            w0, w1, b0, b1 = weights
            for offset in range(0, len(order), batch_size):
                batch = indices[offset:offset + batch_size]
                xb, yb = xt[batch], yt[batch]
                hidden = (xb @ w0 + b0).relu()
                logits = hidden @ w1 + b1
                if binary:
                    delta = logits.sigmoid() - yb[:, None]
                else:
                    delta = logits.softmax(1)
                    delta[torch.arange(len(batch), device=device), yb] -= 1
                dh = (delta @ w1.T) * (hidden > 0)
                n = len(batch)
                grads = [(xb.T @ dh + p["alpha"] * w0) / n,
                         (hidden.T @ delta + p["alpha"] * w1) / n,
                         dh.mean(0), delta.mean(0)]
                step += 1
                lr = 0.001 * np.sqrt(1 - 0.999 ** step) / (1 - 0.9 ** step)
                for param, moment, variance, grad in zip(weights, moments, variances, grads):
                    moment.mul_(0.9).add_(grad, alpha=0.1)
                    variance.mul_(0.999).addcmul_(grad, grad, value=0.001)
                    param.addcdiv_(moment, variance.sqrt().add_(1e-8), value=-lr)
            pred = torch.cat(list(predict_tensor(xv, weights)))
            score = int((pred == yv).sum().item()) / len(yv)
            history.append(score)
            stale = stale + 1 if score < best_score + 1e-4 else 0
            if score > best_score:
                best_score = score
                best = [value.clone() for value in weights]
            if epoch == 0 or (epoch + 1) % 10 == 0 or stale > 10:
                print("%s epoch=%d validation=%.6f seconds=%.1f" %
                      (meta["id"], epoch + 1, score, time.perf_counter() - start),
                      file=sys.stderr, flush=True)
            if stale > 10:
                break
        fit_seconds = time.perf_counter() - start
        del xt, xv, yt, yv, X
        test = load_matrix(directory, "test")
        if test.ndim != 2 or test.shape[1] != weights[0].shape[0] or test.dtype != np.float32:
            raise ValueError("Prepared test feature schema mismatch")
        if not np.isfinite(test).all() or len(test) == 0:
            raise ValueError("Test features must be nonempty and finite")
        xe = torch.tensor(np.asarray(test), device=device)
        pred = torch.cat(list(predict_tensor(xe, best))).cpu().numpy()
        atomic_npy(directory / "predictions.npy", pred)
        atomic_npz(directory / "best_weights.npz",
                   **{key: value.cpu().numpy() for key, value in zip(names, best)})
    return {"epochs": epoch + 1, "validation_scores": history,
            "fit_seconds": fit_seconds, "worker_seconds": time.perf_counter() - start,
            "torch_version": torch.__version__, "device": device,
            "backend": "torch_cpu", "protocol": "sklearn_adam_fp32_cpu_v1",
            "optimizer_updates": step}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("Threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps({"ready": True, "device": "cpu", "threads": args.threads,
                      "torch_version": torch.__version__}), flush=True)
    for line in sys.stdin:
        try:
            command = json.loads(line)
            if set(command) != {"directory"}:
                raise ValueError("Worker command must contain only a prepared job directory")
            result = train(command["directory"])
            print(json.dumps({"ok": True, "result": result}), flush=True)
        except Exception:
            message = traceback.format_exc()
            print(message, file=sys.stderr, flush=True)
            print(json.dumps({"ok": False, "error": message}), flush=True)
        finally:
            gc.collect()


if __name__ == "__main__":
    main()
