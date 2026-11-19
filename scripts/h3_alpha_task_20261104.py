"""Task Scheduler entry point: durable logs, PID/checkpoint, no SSH dependency."""
from __future__ import annotations
import argparse
import contextlib
import os
from pathlib import Path
import runpy
import sys
import traceback

from h3_alpha_io_20261104 import ROOT, checkpoint, utc


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['nts','full'],required=True)
    parser.add_argument('--snapshot')
    args=parser.parse_args()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key]='1'
    os.environ['PYTHONIOENCODING']='utf-8'
    os.environ['PYTHONUNBUFFERED']='1'
    (ROOT/'runtime').mkdir(exist_ok=True,parents=True)
    script='build_h3_nts_q_20261104.py' if args.stage=='nts' else 'run_h3_alpha_20261104.py'
    with (ROOT/'runtime'/f'{args.stage}.stdout.log').open('a',encoding='utf-8',buffering=1) as output:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            print('INVOCATION',utc(),'PID',os.getpid(),'STAGE',args.stage,flush=True)
            try:
                checkpoint('TASK_STARTED',task_stage=args.stage,script=script)
                sys.argv=[str(Path(__file__).with_name(script))]
                if args.snapshot:
                    if args.stage!='full':raise ValueError('--snapshot applies only to the alpha stage')
                    sys.argv+=['--snapshot',args.snapshot]
                runpy.run_path(sys.argv[0],run_name='__main__')
            except BaseException as exc:
                checkpoint('TASK_FAILED',task_stage=args.stage,error=repr(exc))
                traceback.print_exc()
                raise
