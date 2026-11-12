"""Doc 03 §14 item 48: apply each no-patch run's budget-refused edits to the base snapshot, one results dir per run.

Then, with Docker: lab.eval.verify_patches --results <dir> for every dir with a non-empty patch.
Usage: python -m lab.analysis.refused_edits data/analysis/refused_edits_cf"""
import json,subprocess,sys,tarfile,tempfile
from pathlib import Path
from lab.analysis.near_miss import load_runs,bucket
from lab.eval.verifiable import load_verifiable
V=load_verifiable(); out=Path(sys.argv[1]); out.mkdir(exist_ok=True)
runs=[r for r in load_runs(Path('data/eval')) if bucket(r)=='no_patch' and V.get(r['task']) and r['edits_after_budget']>0]
print(len(runs))
man=[]
for r in runs:
    t=json.loads(Path(f"data/eval/{r['session']}/results/{r['arm']}/traces/trace_{r['task']}.json").read_text())
    eds=[]
    for st in t['steps']:
        for tc in st.get('tool_calls') or []:
            if tc.get('function_name') in('edit_file','write_file') and 'BudgetExceeded' in str((st.get('observation') or {}).get('content')):
                eds.append(tc)
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(f"data/competition/snapshots/{r['task']}.tgz") as tf: tf.extractall(d,filter='data')
        root=Path(d); kids=[k for k in root.iterdir()]
        if len(kids)==1 and kids[0].is_dir(): root=kids[0]
        if not (root/'.git').exists():
            subprocess.run('git init -q && git add -A -f . && git -c user.name=x -c user.email=x@x commit -qm b',shell=True,cwd=root,check=True)
        res=[]
        for tc in eds:
            a=tc['arguments']; 
            if tc['function_name']!='edit_file' or not a.get('filepath'): res.append('skip'); continue
            f=root/a['filepath'].removeprefix('/workspace/').lstrip('/')
            if not f.exists(): res.append('nofile'); continue
            s=f.read_text()
            if a.get('old_string') and s.count(a['old_string'])>=1:
                f.write_text(s.replace(a['old_string'],a['new_string'],1)); res.append('ok')
            else: res.append('nomatch')
        diff=subprocess.run(['git','diff'],cwd=root,capture_output=True,text=True).stdout
    name=f"{r['session']}__{r['arm']}__{r['task']}"
    dd=out/name; (dd/'patches').mkdir(parents=True,exist_ok=True)
    (dd/'patches'/f"{r['task']}.patch").write_text(diff)
    man.append(dict(name=name,task=r['task'],session=r['session'],arm=r['arm'],edits=res,patch=len(diff)))
    print(name,res,len(diff))
json.dump(man,open(out/'manifest.json','w'),indent=1)
