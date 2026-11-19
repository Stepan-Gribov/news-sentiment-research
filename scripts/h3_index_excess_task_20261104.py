"""Scheduled entry point: pin/copy every executable before production imports."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback

ROOT=Path('./analysis_outputs/h3_index_excess_20261104')
CODE=Path(__file__).parent
SPEC_SHA='b0d982e5bf57866a2ea6d0fca5e57186a54076d4359f5ebe42ec752224dbd74b'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
    config_path=ROOT/'runtime/launch_config.json';config=json.loads(config_path.read_text(encoding='utf-8'))
    if sha(CODE/'h3_index_excess_spec_20261104.json')!=SPEC_SHA:raise AssertionError('Frozen spec mismatch')
    names=['run_h3_index_excess_20261104.py','h3_index_excess_core_20261104.py','fit_h3_index_excess_fe_20261104.py',
        'h3_alpha_core_20261104.py','h3_alpha_io_20261104.py','test_h3_index_excess_core_20261104.py','h3_index_excess_spec_20261104.json']
    sources={name:rec(CODE/name) for name in names}
    validated={}
    for name,r in config['inputs'].items():
        if sha(r['path'])!=r['sha256']:raise AssertionError('Input mismatch '+name)
        validated[name]=rec(r['path'])
    version=hashlib.sha256(json.dumps(dict(config=config,sources=sources),sort_keys=True).encode()).hexdigest()[:16]
    jobdir=ROOT/'jobs'/version;frozen=jobdir/'code';frozen.mkdir(parents=True,exist_ok=True)
    snapshots={}
    for name,r in sources.items():
        dest=frozen/name
        if not dest.exists():shutil.copy2(r['path'],dest)
        if sha(dest)!=r['sha256']:raise AssertionError('Immutable code snapshot conflict '+name)
        snapshots[name]=rec(dest)
    inputs=validated
    for key in ['benchmark_audit','benchmark_manifest']:
        r=inputs[key];dest=jobdir/(key+'.json')
        if not dest.exists():shutil.copy2(r['path'],dest)
        if sha(dest)!=r['sha256']:raise AssertionError('Immutable source evidence snapshot conflict')
        inputs[key]=rec(dest)
    job=dict(config,version=version,inputs=inputs,code=snapshots)
    jobpath=jobdir/'job.json'
    if jobpath.exists():
        if json.loads(jobpath.read_text(encoding='utf-8'))!=job:raise AssertionError('Immutable job conflict')
    else:save(jobpath,job)
    save(ROOT/'runtime/job_ready.json',dict(version=version,job=str(jobpath),job_sha256=sha(jobpath),output=str(ROOT/'outputs'/version)))
    if (ROOT/'outputs'/version/'manifest.json').exists():print('Already complete, no duplicate',flush=True);return
    tests={}
    for name in ['test_h3_index_excess_core_20261104.py','fit_h3_index_excess_fe_20261104.py']:
        result=subprocess.run([sys.executable,str(frozen/name)],capture_output=True,text=True,encoding='utf-8',check=True)
        test=json.loads(result.stdout)
        if test['status']!='PASS':raise AssertionError('Synthetic contract failed')
        tests[name]=test
    save(jobdir/'synthetic_QA.json',tests)
    subprocess.run([sys.executable,'-u',str(frozen/'run_h3_index_excess_20261104.py'),'--job',str(jobpath)],check=True)


if __name__=='__main__':
    ROOT.joinpath('runtime').mkdir(parents=True,exist_ok=True)
    with (ROOT/'runtime/full.stdout.log').open('a',encoding='utf-8',buffering=1) as log:
        try:
            # subprocess inherits these handles; do not depend on the SSH tty.
            import contextlib
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                sys.stdout=log;sys.stderr=log
                # Pass explicit handles through Popen's defaults via file descriptors.
                original_run=subprocess.run
                def logged_run(*args,**kwargs):
                    if not kwargs.get('capture_output'):kwargs.update(stdout=log,stderr=log)
                    return original_run(*args,**kwargs)
                subprocess.run=logged_run
                main()
        except BaseException as e:
            save(ROOT/'runtime/task_failed.json',dict(error=repr(e),pid=os.getpid()))
            traceback.print_exc(file=log);raise
