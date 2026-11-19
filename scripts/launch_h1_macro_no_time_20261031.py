"""Detached scheduled-job entry point; research execution remains on research-host."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / 'plan.json'
EXPECTED = '3e0a167a6ac0fcb6958d346c9d7e8dcded134d98827eb8f5a6841a30a270f383'


def main():
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Launch plan checksum mismatch')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    for name, expected in plan['frozen_code_sha256'].items():
        if hashlib.sha256((ROOT / 'code' / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen implementation changed: ' + name)
    if (ROOT / 'completion.json').exists() or (ROOT / 'run.lock').exists():
        raise RuntimeError('A completed or active run already exists')
    command = ['./local/research-data-service/venv/Scripts/python.exe',
               str(ROOT / 'code/run_h1_macro_no_time_20261031.py'),
               '--plan', str(PLAN), '--plan-sha', EXPECTED]
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[key] = '4'
    with (ROOT / 'run.stdout.log').open('w', encoding='utf-8') as stdout, \
            (ROOT / 'run.stderr.log').open('w', encoding='utf-8') as stderr:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr)
        record = dict(pid=process.pid, launcher_pid=os.getpid(), plan_sha256=EXPECTED,
                      started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        (ROOT / 'job.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        record['exit_code'] = process.wait()
        record['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        (ROOT / 'job.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
