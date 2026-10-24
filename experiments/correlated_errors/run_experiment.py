#!/usr/bin/env python3
"""Synthetic benchmark of aggregation under a shared-error mixture."""
import argparse, csv, json, random
from pathlib import Path
SKILLS=(0.80,0.70,0.62)
Q_VALUES=(0.0,0.25,0.5,0.75,1.0)

def make_rows(seed, q, n, inflation=0.10):
    rng=random.Random(seed*1000003 + int(q*1000)*9176)
    rows=[]
    for _ in range(n):
        y=rng.randint(0,1); shared=rng.random()<q; common=rng.random()
        agents=[]
        for skill in SKILLS:
            ok=(common<skill) if shared else (rng.random()<skill)
            pred=y if ok else 1-y
            conf=max(0.51,min(0.99,skill+inflation+rng.uniform(-0.08,0.08)))
            agents.append((pred,conf,int(ok)))
        rows.append((y,agents))
    return rows

def ece(labels, probs, bins=10):
    n=len(labels); total=0.0
    for b in range(bins):
        ix=[i for i,p in enumerate(probs) if b/bins<=p<(b+1)/bins or (b==bins-1 and p==1)]
        if ix: total += len(ix)/n*abs(sum(probs[i] for i in ix)/len(ix)-sum(labels[i] for i in ix)/len(ix))
    return total

def score(rows, policy, correction=0.0):
    labels=[]; probs=[]
    for y,agents in rows:
        labels.append(y)
        if policy=='majority':
            probs.append(float(sum(p for p,_,_ in agents)>len(agents)/2))
        else:
            weights=[max(.01,min(.99,c-(correction if policy=='calibrated_weighted' else 0.0))) for _,c,_ in agents]
            probs.append(sum(p*w for (p,_,_),w in zip(agents,weights))/sum(weights))
    return {'accuracy':sum((p>=.5)==bool(y) for y,p in zip(labels,probs))/len(labels),
            'brier':sum((p-y)**2 for y,p in zip(labels,probs))/len(labels),
            'ece_10bin':ece(labels,probs),'n':len(labels)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=5); ap.add_argument('--scenarios',type=int,default=2500); ap.add_argument('--out',default='outputs'); a=ap.parse_args()
    if a.seeds<1 or a.scenarios<1: ap.error('--seeds and --scenarios must be positive')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); results=[]
    for q in Q_VALUES:
      for seed in range(a.seeds):
        calibration=make_rows(seed*2+11,q,max(500,a.scenarios//2))
        test=make_rows(seed*2+29,q,a.scenarios)
        diffs=[c-ok for _,agents in calibration for _,c,ok in agents]
        correction=max(-.25,min(.25,sum(diffs)/len(diffs)))
        for policy in ('majority','raw_weighted','calibrated_weighted'):
          results.append({'q':q,'seed':seed,'policy':policy,'correction':correction,**score(test,policy,correction)})
    with (out/'metrics_summary.csv').open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    manifest={'experiment':'correlated-errors','synthetic_only':True,'seeds':a.seeds,'scenarios_per_seed':a.scenarios,'skills':SKILLS,'q_values':Q_VALUES,'q_interpretation':'shared-shock mixture probability, not pairwise correlation'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f"Wrote {len(results)} summary rows to {out/'metrics_summary.csv'} and manifest.")
if __name__=='__main__': main()
