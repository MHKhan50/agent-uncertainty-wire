#!/usr/bin/env python3
"""Synthetic confidence-inflation experiment with held-out calibration."""
import argparse,csv,json,random
from pathlib import Path
SKILLS=(.80,.70,.62); LEVELS=(0,.05,.10,.15,.20,.25)

def make_split(seed,n,inflation):
 rng=random.Random(seed); rows=[]
 for _ in range(n):
  y=rng.randint(0,1); agents=[]
  for skill in SKILLS:
   ok=rng.random()<skill; pred=y if ok else 1-y
   conf=max(.51,min(.99,skill+inflation+rng.uniform(-.08,.08)))
   agents.append((pred,conf,int(ok)))
  rows.append((y,agents))
 return rows

def correction_from(calibration):
 vals=[conf-ok for _,agents in calibration for _,conf,ok in agents]
 return max(-.25,min(.25,sum(vals)/len(vals)))

def metrics(rows,correction=0.0,adjust=False):
 ys=[]; ps=[]
 for y,agents in rows:
  weights=[max(.01,min(.99,c-correction if adjust else c)) for _,c,_ in agents]
  ys.append(y); ps.append(sum(pred*w for (pred,_,_),w in zip(agents,weights))/sum(weights))
 ece=0.0
 for b in range(10):
  ix=[i for i,p in enumerate(ps) if b/10<=p<(b+1)/10 or (b==9 and p==1)]
  if ix: ece+=len(ix)/len(ys)*abs(sum(ps[i] for i in ix)/len(ix)-sum(ys[i] for i in ix)/len(ix))
 return {'accuracy':sum((p>=.5)==bool(y) for y,p in zip(ys,ps))/len(ys),'brier':sum((p-y)**2 for y,p in zip(ys,ps))/len(ys),'ece_10bin':ece,'n':len(ys)}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=10); ap.add_argument('--calibration-size',type=int,default=1000); ap.add_argument('--test-size',type=int,default=2000); ap.add_argument('--out',default='outputs'); a=ap.parse_args()
 if min(a.seeds,a.calibration_size,a.test_size)<1: ap.error('all sizes and seeds must be positive')
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); result=[]
 for level in LEVELS:
  for seed in range(a.seeds):
   cal=make_split(seed*2+1001,a.calibration_size,level); test=make_split(seed*2+2003,a.test_size,level); corr=correction_from(cal)
   for method,adjust in (('raw_confidence',False),('calibration_corrected',True)):
    result.append({'inflation':level,'seed':seed,'method':method,'estimated_correction':corr,**metrics(test,corr,adjust)})
 with (out/'metrics_summary.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(result[0])); w.writeheader(); w.writerows(result)
 manifest={'experiment':'confidence-inflation','synthetic_only':True,'seeds':a.seeds,'calibration_size':a.calibration_size,'test_size':a.test_size,'inflation_levels':LEVELS}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 print(f"Wrote {len(result)} summary rows to {out/'metrics_summary.csv'} and manifest.")
if __name__=='__main__': main()
