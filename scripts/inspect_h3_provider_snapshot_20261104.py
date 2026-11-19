"""Inspect a small approved public documentary snapshot, not a live collector."""
import json
from pathlib import Path
from h3_alpha_io_20261104 import ROOT,file_record
p=ROOT/'research_tr_evidence_arsagera/cached_monthly_documentary_snapshot_20261104.json'
file_record(p,'cb3054b3a1305a81d31dc1078caeef4d24489a821e6d24dd9625907b5190e4fd')
d=json.loads(p.read_text(encoding='utf-8'))
print(json.dumps({'type':type(d).__name__,'keys':list(d) if isinstance(d,dict) else None,
    'contents':{k:({'items':len(v),'first':v[0] if v else None,'last':v[-1] if v else None}
       if isinstance(v,list) else v) for k,v in d.items()} if isinstance(d,dict) else {'items':len(d),'first':d[0],'last':d[-1]}},ensure_ascii=False,indent=2))
