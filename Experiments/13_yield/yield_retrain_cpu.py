"""CPU continuous-yield extension MLP; test outcomes never reach this worker."""
import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(directory, job):
    for name in ('fit.npz', 'test.npy', 'preprocessing.npz'):
        if sha(directory / name) != job['inputs'][name]:
            raise ValueError('Changed worker input: ' + name)
    with np.load(directory / 'fit.npz', allow_pickle=False) as z:
        if set(z.files) != {'X_train', 'X_val', 'y_train', 'y_val'}:
            raise ValueError('Worker accepts only training and validation outcomes')
        a = {k: z[k] for k in z.files}
    test = np.load(directory / 'test.npy', allow_pickle=False)
    for part in ('train', 'val'):
        x, y = a['X_' + part], a['y_' + part]
        if x.ndim != 2 or not len(x) or y.shape != (len(x),):
            raise ValueError('Invalid regression input shape')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite regression input')
    if (test.ndim != 2 or not len(test) or not np.isfinite(test).all()
            or a['X_train'].shape[1] != a['X_val'].shape[1]
            or test.shape[1] != a['X_train'].shape[1]):
        raise ValueError('Invalid prediction features')
    return a, test


def train(directory):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU development worker requires CUDA hidden')
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = Path(directory)
    job = json.loads((directory / 'job.json').read_text())
    a, test = inputs(directory, job)
    p = job['params']
    torch.manual_seed(p['seed'])
    device = torch.device('cpu')
    xt, xv = [torch.tensor(a['X_' + s], dtype=torch.float32, device=device) for s in ('train', 'val')]
    yt, yv = [torch.tensor(a['y_' + s], dtype=torch.float32, device=device) for s in ('train', 'val')]
    model = nn.Sequential(nn.Linear(xt.shape[1], p['hidden']), nn.ReLU(), nn.Linear(p['hidden'], 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    mse = nn.MSELoss()
    best, stopping, stale, history = float('inf'), float('inf'), 0, []
    start = time.perf_counter()
    for epoch in range(1, p['max_epochs'] + 1):
        model.train()
        order = torch.randperm(len(xt), device=device)
        for offset in range(0, len(xt), p['batch_size']):
            ix = order[offset:offset + p['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = mse(model(xt[ix]).squeeze(-1), yt[ix])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(mse(model(xv).squeeze(-1), yv))
        if not np.isfinite(value):
            raise ValueError('Nonfinite validation MSE')
        history.append({'epoch': epoch, 'validation_mse_standardized': value})
        if value < best:
            best, selected, state = value, epoch, copy.deepcopy(model.state_dict())
        if value < stopping - p['min_delta']:
            stopping, stale = value, 0
        else:
            stale += 1
        if stale >= p['patience']:
            break
    model.load_state_dict(state)
    model.eval()
    with np.load(directory / 'preprocessing.npz', allow_pickle=False) as z:
        mean, scale = float(z['target_mean']), float(z['target_scale'])
    pred = []
    with torch.no_grad():
        for start_row in range(0, len(test), 4096):
            x = torch.tensor(test[start_row:start_row + 4096], dtype=torch.float32, device=device)
            pred.append(model(x).squeeze(-1).cpu().numpy())
    pred = np.concatenate(pred).astype(np.float64) * scale + mean
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite yield predictions')
    np.save(directory / 'predictions.npy', pred)
    torch.save({'state_dict': {k: v.cpu() for k, v in state.items()}, 'params': p,
                'input_dim': int(xt.shape[1]), 'selected_epoch': selected,
                'plan_sha256': job['plan_sha256'], 'job_sha256': sha(directory / 'job.json')},
               directory / 'best_model.pt')
    receipt = {'history': history, 'epochs': epoch, 'selected_epoch': selected,
               'best_validation_mse_standardized': best, 'seconds': time.perf_counter() - start,
               'physical_gpu': None, 'execution_device': 'cpu', 'continuous_yield_extension': True, 'torch_version': torch.__version__,
               'worker_sha256': sha(__file__), 'job_sha256': sha(directory / 'job.json'),
               'artifacts': {n: sha(directory / n) for n in ('predictions.npy', 'best_model.pt')}}
    (directory / 'training.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'dataset': job['dataset'], 'split': job['split'], 'fingerprint': job['fingerprint'],
                      'epochs': epoch, 'seconds': receipt['seconds']}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    train(ap.parse_args().directory)
