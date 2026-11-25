import json,glob,os,sys,collections
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from reverify_base import *
B=json.load(open(S+'/base.json'));ok={t for t in B if B[t]['gold']==0 and B[t]['empty']!=0}
V=json.load(open(R+'/data/analysis/verifiable_tasks.json'))
runs="pool-gov-gpu pool-mix-gpu pool-tb-gpu pool-tb22-gpu pool-tbslec-gpu pool-to-gpu pool-tont-gpu pool-e4b-gpu".split()
rows=[]
for r in runs:
  for d in sorted(glob.glob(f'{R}/data/eval/{r}/results/*')):
    f=d+'/task_results.jsonl'
    if not os.path.exists(f):continue
    for l in open(f):
      x=json.loads(l);t=x['instance_id'];p=f'{d}/patches/{t}.patch'
      rows.append(dict(pool=r,arm=os.path.basename(d),t=t,h=bool(x['resolved']),err=x.get('error'),patch=open(p).read() if os.path.exists(p) else ''))
keys={(x['t'],x['patch']) for x in rows if x['patch'].strip() and x['t'] in ok}
print(len(rows),len(keys))
def one(k):
  try:return k,run(*k)
  except Exception as e:return k,('exc',str(e))
with ThreadPoolExecutor(5) as p:res=dict(p.map(one,keys))
for x in rows:
  k=(x['t'],x['patch']); x['v']=(res[k][0] if k in res else None); x['tail']=res[k][1][-900:] if k in res else ''
json.dump(rows,open(S+'/rv_rows.json','w'))
def show(name,sel):
  n=len(sel);print(name,n,'harness',sum(x['h'] for x in sel),'reparado',sum(x['v']==0 for x in sel))
m=[x for x in rows if x['t'] in ok]
show('31B ok',[x for x in m if 'e4b' not in x['pool']]);show('E4B ok',[x for x in m if 'e4b' in x['pool']])
print('codes',collections.Counter(str(x['v']) for x in m))
print('h y no v:',collections.Counter(x['t'] for x in m if x['h'] and x['v']!=0)); print('v y no h',collections.Counter(x['t'] for x in m if x['v']==0 and not x['h']))
c=collections.defaultdict(list)
for x in m: c[('E4B' if 'e4b' in x['pool'] else '31B',x['t'])].append(x)
for k,v in sorted(c.items()): print(k,'n',len(v),'h',sum(x['h'] for x in v),'v',sum(x['v']==0 for x in v),'empty',sum(not x['patch'].strip() for x in v),'tmo',sum('imeout' in str(x['err']) for x in v))
c=collections.defaultdict(list)
for x in m: c[(x['pool'],x['arm'])].append(x)
for k,v in sorted(c.items()): print(k,len(v),'h',sum(x['h'] for x in v),'v',sum(x['v']==0 for x in v))
