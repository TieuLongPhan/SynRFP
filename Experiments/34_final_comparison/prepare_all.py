"""Prepare the three new components without changing historical producers."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path.home() / '.local/share/synrfp-runs/final_comparison_v1'
spec = importlib.util.spec_from_file_location('final_prepare_budget', HERE.parent / '25_followup_scheduler/budget.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Preparation must hide CUDA')
    manifest = json.loads((ROOT / 'execution.json').read_text())
    for path, digest in manifest['sources_sha256'].items():
        if B.sha(path) != digest:
            raise ValueError('Final-comparison source changed: ' + path)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_*.py', '-v'], check=True)
    for component in ('report', 'regression', 'clustering'):
        subprocess.run([sys.executable, '-u', str(HERE / (component + '.py')), '--prepare'], check=True)
    subprocess.run([sys.executable, '-u', str(HERE / 'report.py'), '--once'], check=True)
    plans = {str(ROOT / component / 'plan.json'): B.sha(ROOT / component / 'plan.json')
             for component in ('report', 'regression', 'clustering')}
    for path, digest in manifest['sources_sha256'].items():
        if B.sha(path) != digest:
            raise ValueError('Final-comparison source changed during preparation')
    B.write(ROOT / 'PREPARED.json', dict(phase='complete', plan_sha256=plans,
        execution_sha256=B.sha(ROOT / 'execution.json'), prepared_utc=B.now(), fits_launched=0))


if __name__ == '__main__':
    main()
