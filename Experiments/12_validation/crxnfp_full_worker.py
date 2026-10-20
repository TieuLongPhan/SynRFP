#!/usr/bin/env python
"""Persistent CUDA worker. Python 3.8 compatible; numpy/torch dependencies only.

Reads JSON commands on stdin and writes one JSON response per job on stdout.
Training progress goes to stderr. Prepared inputs come from mlp_gpu_queue.py.
"""
import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


def train(directory, device):
    directory = Path(directory)
    meta = json.loads((directory / 'job.json').read_text())
    p = meta['params']
    with np.load(directory / 'state.npz') as z:
        arrays = {k: z[k] for k in z.files}
    start = time.perf_counter()
    X = np.load(directory / 'train.npy', mmap_mode='r')
    xt = torch.tensor(X[arrays['train']], device=device)
    xv = torch.tensor(X[arrays['val']], device=device)
    yt = torch.tensor(arrays['y'][arrays['train']], device=device)
    yv = torch.tensor(arrays['y'][arrays['val']], device=device)
    weights = [torch.tensor(arrays[k], device=device) for k in ['w0', 'w1', 'b0', 'b1']]
    binary = weights[1].shape[1] == 1
    moments = [torch.zeros_like(t) for t in weights]
    variances = [torch.zeros_like(t) for t in weights]
    rng = np.random.RandomState()
    rng.set_state(('MT19937', arrays['rng_keys'], int(arrays['rng_pos']),
                   int(arrays['rng_has_gauss']), float(arrays['rng_gauss'])))
    order = np.arange(len(xt))
    best_score, stale, step = -np.inf, 0, 0
    history = []
    batch_size = min(p['batch_size'], len(xt))

    def predict_tensor(data, w):
        w0, w1, b0, b1 = w
        for offset in range(0, len(data), 8192):
            logits = (data[offset:offset + 8192] @ w0 + b0).relu() @ w1 + b1
            yield (logits[:, 0] > 0).long() if binary else logits.argmax(1)

    with torch.no_grad():
        for epoch in range(p['max_iter']):
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
                grads = [(xb.T @ dh + p['alpha'] * w0) / n,
                         (hidden.T @ delta + p['alpha'] * w1) / n,
                         dh.mean(0), delta.mean(0)]
                step += 1
                lr = 0.001 * np.sqrt(1 - 0.999 ** step) / (1 - 0.9 ** step)
                for param, m, v, grad in zip(weights, moments, variances, grads):
                    m.mul_(0.9).add_(grad, alpha=0.1)
                    v.mul_(0.999).addcmul_(grad, grad, value=0.001)
                    param.addcdiv_(m, v.sqrt().add_(1e-8), value=-lr)
            pred = torch.cat(list(predict_tensor(xv, weights)))
            # Integer count avoids float32 rounding in sklearn's accuracy comparison.
            score = int((pred == yv).sum().item()) / len(yv)
            history.append(score)
            stale = stale + 1 if score < best_score + 1e-4 else 0
            if score > best_score:
                best_score = score
                best = [t.clone() for t in weights]
            if epoch == 0 or (epoch + 1) % 10 == 0 or stale > 10:
                print('%s epoch=%d validation=%.6f seconds=%.1f' %
                      (meta['id'], epoch + 1, score, time.perf_counter() - start),
                      file=sys.stderr, flush=True)
            if stale > 10:
                break
        torch.cuda.synchronize(device)
        fit_seconds = time.perf_counter() - start
        # Release training tensors before loading test features.
        del xt, xv, yt, yv, X
        xe = torch.tensor(np.load(directory / 'test.npy'), device=device)
        pred = torch.cat(list(predict_tensor(xe, best))).cpu().numpy()
        np.save(directory / 'predictions.npy', pred)
        # Save the unchanged best weights locally for reuse and prediction replay.
        np.savez_compressed(directory / 'best_weights.npz',
                            **{k: t.cpu().numpy() for k, t in zip(['w0', 'w1', 'b0', 'b1'], best)})
    return dict(epochs=epoch + 1, validation_scores=history,
                fit_seconds=fit_seconds, gpu_seconds=time.perf_counter() - start,
                torch_version=torch.__version__, device=device,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    device = 'cuda:%d' % args.device
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.zeros(1, device=device)
    print(json.dumps(dict(ready=True, device=device, gpu=torch.cuda.get_device_name(device))), flush=True)
    for line in sys.stdin:
        try:
            command = json.loads(line)
            torch.cuda.reset_peak_memory_stats(device)
            result = train(command['directory'], device)
            print(json.dumps(dict(ok=True, result=result)), flush=True)
        except Exception:
            message = traceback.format_exc()
            print(message, file=sys.stderr, flush=True)
            print(json.dumps(dict(ok=False, error=message)), flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
