"""Durable scheduled-task entry point for the new relaxed-news root only."""
import argparse
import contextlib
import os
from pathlib import Path
import runpy
import sys
import traceback
from build_h3_relaxed_panel_20261104 import ROOT,checkpoint

p=argparse.ArgumentParser();p.add_argument('--stage',choices=['panel','full'],required=True);args=p.parse_args()
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
ROOT.joinpath('runtime').mkdir(parents=True,exist_ok=True)
script='build_h3_relaxed_panel_20261104.py' if args.stage=='panel' else 'run_h3_relaxed_news_20261104.py'
with (ROOT/'runtime'/(args.stage+'.stdout.log')).open('a',encoding='utf-8',buffering=1) as f:
    with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        try:
            checkpoint('TASK_STARTED',task_stage=args.stage)
            sys.argv=[str(Path(__file__).with_name(script))]
            runpy.run_path(sys.argv[0],run_name='__main__')
        except BaseException as error:
            checkpoint('TASK_FAILED',task_stage=args.stage,error=repr(error));traceback.print_exc();raise
