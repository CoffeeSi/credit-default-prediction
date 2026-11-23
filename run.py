"""Single-command pipeline reproduction; every stage fails immediately on error."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-training', action='store_true', help='Reuse supplied frozen models for evaluation/demo/deck')
    args = parser.parse_args()
    commands = []
    if not args.skip_training:
        commands += [['-m','src.data','download'], ['-m','src.preprocessing','prepare'],
                     ['-m','src.data','eda'], ['-m','src.classical'],
                     ['-m','src.neural','train'], ['-m','src.freeze']]
    commands += [['-m','unittest','discover','-s','tests'], ['-m','src.evaluate'],
                 ['-m','src.demo','--index','0','--reveal-label'], ['scripts/build_presentation.py']]
    env = dict(os.environ, PYTHONHASHSEED='42', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLBACKEND='Agg')
    start = time.perf_counter()
    for command in commands:
        print('\nRunning: '+sys.executable+' '+' '.join(command), flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)
    print(f'Pipeline completed in {time.perf_counter()-start:.1f} seconds.', flush=True)

if __name__ == '__main__':
    main()
