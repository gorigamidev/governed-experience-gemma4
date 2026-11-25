import json,glob,os,re,subprocess,tempfile,tarfile,collections,sys
from concurrent.futures import ThreadPoolExecutor
import tempfile as _t; S=os.environ['RV_WORK']; R=os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
T={json.loads(l)['instance_id']:json.loads(l) for l in open(R+'/data/competition/tasks.jsonl')}
py={'fastapi':'vf','rich':'vr','requests':'vq'}
def ap(root,p):
    for c in (['git','apply','--whitespace=nowarn'],['patch','-p1','-f','-s']):
        r=subprocess.run(c,cwd=root,input=p if p.endswith('\n') else p+'\n',text=True,capture_output=True)
        if r.returncode==0:return True
    return False
def run(t,patch):
    x=T[t];d=tempfile.mkdtemp(dir=S+'/rv')
    with tarfile.open(f'{R}/data/competition/snapshots/{t}.tgz') as a:a.extractall(d,filter='data')
    sub=[q for q in os.listdir(d)]
    root=d if len(sub)!=1 else os.path.join(d,sub[0])
    subprocess.run(['git','init','-q'],cwd=root)
    try:
        if patch.strip() and not ap(root,patch):return 'noapply',''
        if not ap(root,x['test_patch']):return 'notest',''
        files=[f for f in re.findall(r'^\+\+\+ b/(\S+)',x['test_patch'],re.M) if f.endswith('.py') and os.path.basename(f).startswith('test')]
        env={'PATH':os.environ['PATH'],'HOME':d,'PYTHONPATH':f'{root}:{root}/src','LANG':'en_US.UTF-8','PYTHONDONTWRITEBYTECODE':'1'}
        v=py[t.split('_')[0]]
        r=subprocess.run([f'{S}/rv/{v}/bin/python','-m','pytest','-q','-x' if 0 else '-q','--timeout=120','-p','no:cacheprovider','-W','ignore','-o','addopts=','-o','filterwarnings=',*files],cwd=root,env=env,capture_output=True,text=True,timeout=600)
        return r.returncode,r.stdout
    except subprocess.TimeoutExpired:return 'timeout',''
    finally: subprocess.run(['rm','-rf',d])
def base(t):
    try:return t,run(t,T[t]['patch']),run(t,'')
    except Exception as e:return t,('exc',str(e)),('exc','')
if __name__=='__main__':
    ids=sorted(t for t in T if os.path.exists(f'{R}/data/competition/snapshots/{t}.tgz') and t.split('_')[0] in py)
    with ThreadPoolExecutor(5) as p:res=list(p.map(base,ids))
    out={}
    for t,g,e in res:
        # fail set comparison: failing test ids
        f=lambda o:set(re.findall(r'^(?:FAILED|ERROR) (\S+)',o,re.M))
        out[t]={'gold':g[0],'empty':e[0],'gf':sorted(f(g[1])),'ef':sorted(f(e[1])),'tail':g[1][-600:]}
    json.dump(out,open(S+'/base.json','w'))
    V=json.load(open(R+'/data/analysis/verifiable_tasks.json'))
    ok=[t for t in out if out[t]['gold']==0 and out[t]['empty'] not in(0,)]
    print(len(out),'ok strict',len(ok),collections.Counter(t.split('_')[0] for t in ok))
    print('already verifiable but not here',[t for t in V if V[t] and t in out and t not in ok])
    print('new',[t for t in ok if not V.get(t)])
    for t in out:
        if t not in ok: print(t,out[t]['gold'],out[t]['empty'],len(out[t]['gf']),len(out[t]['ef']),out[t]['tail'][-160:].replace('\n',' | '))
