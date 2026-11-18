"""Compare boundary action classifiers, expected gains, and agreement policies."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

spec=importlib.util.spec_from_file_location('p20',Path(__file__).resolve().parents[1]/'P20/run.py')
p20=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p20)
p=argparse.ArgumentParser()
p.add_argument('dataset',type=Path)
p.add_argument('p20',type=Path)
p.add_argument('out',type=Path)
args=p.parse_args()
args.out.mkdir(parents=True)
start=time.monotonic()
rng=np.random.default_rng(20)
clips=p20.load_manifest(args.dataset)
p20.dump(args.out/'protocol.json',dict(dataset=str(args.dataset),
    dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),p20=str(args.p20),
    features=p20.FEATURES,train_sampling='P20 RNG, at most 700 GT candidates per frame',
    classification='balanced classes as in P20',regression='metric gain from each side, divided by scaled raw depth',
    threshold_search='keep all; score > 0; top-score quantiles from 1% through 50% of valid action proposals',
    ensembles=['SVM+GBDT','RF+GBDT','Logistic+GBDT'],
    selection='min scene-macro candidate MAE on validation; test never used for fitting or tuning',
    matching='test diagnostic: match P20 edit count when possible; also compare frozen P20 and best ensemble at their smaller per-clip budget with P20 ranked by raw confidence',
    actions=['keep','near','far']))

train=[]
evaluation={}
for clip in clips:
    frames,scale,_=p20.candidates(clip)
    name=Path(clip['cache']).parent.name
    existing=None
    if clip['split']!='train':
        existing=np.load(args.p20/name/'depths.npz')
    rows=[]
    for f in frames:
        mask=np.flatnonzero(f['test'])
        if existing is None and len(mask)>700:
            mask=rng.choice(mask,700,replace=False)
        x=f['features'][mask]
        e=f['error'][mask]
        oracle=f['oracle'][mask]
        if existing is None:
            camera_z=f['depths'][mask,0]*scale
            gains=(e[:,:1]-e[:,1:])/camera_z[:,None]
            train.append((x,oracle,gains))
        else:
            i=f['frame']
            y,u=f['y'][mask],f['x'][mask]
            assert existing['candidate'][i,y,u].all()
            rows.append((x,e,existing['action'][i,y,u]))
    if existing is not None:
        existing.close()
        evaluation[name]=dict(scene=clip['scene'],split=clip['split'],domain=clip['domain'],
                              arrays=tuple(np.concatenate(col) for col in zip(*rows)))
    print('features',name,flush=True)

x,y,gain=map(np.concatenate,zip(*train))
ceiling=min(2500,min(int((y==k).sum()) for k in range(3)))
idx=np.concatenate([rng.choice(np.flatnonzero(y==k),ceiling,replace=False) for k in range(3)])
classifiers={
    'RF': RandomForestClassifier(n_estimators=160,max_depth=13,min_samples_leaf=12,n_jobs=4,random_state=20),
    'Logistic': make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,random_state=20)),
}
for model in classifiers.values():
    model.fit(x[idx],y[idx])
regressors=[HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,min_samples_leaf=40,
             l2_regularization=1,early_stopping=False,random_state=20).fit(x,gain[:,j]) for j in range(2)]
joblib.dump(dict(classifiers=classifiers,regressors=regressors),args.out/'models.joblib')
names=['GBDT','RF','Logistic','SVM+GBDT','RF+GBDT','Logistic+GBDT']
for entry in evaluation.values():
    features,e,svm=entry['arrays']
    g=np.column_stack([m.predict(features) for m in regressors])
    direction=g.argmax(1)+1
    score=g.max(1)
    proposed={'GBDT':(direction,score)}
    for name,model in classifiers.items():
        prob=model.predict_proba(features)
        predicted=prob.argmax(1)
        proposed[name]=(predicted,prob[:,1:].max(1)-prob[:,0])
    for name,other in [('SVM+GBDT',svm),('RF+GBDT',proposed['RF'][0]),('Logistic+GBDT',proposed['Logistic'][0])]:
        proposed[name]=(np.where(other==direction,direction,0),score)
    entry.update(error=e,svm=svm,proposed=proposed,svm_priority=-features[:,4])
    del entry['arrays']

val=[q for q in evaluation.values() if q['split']=='val']

def apply(action,score,threshold):
    return np.where(score>threshold,action,0)

def metric(q,action):
    e=q['error']
    after=e[np.arange(len(e)),action]
    before=e[:,0]
    return dict(pixels=len(e),edited=int((action!=0).sum()),MAE_cm=float(after.mean(dtype=float)*100),
                improved=int((after<before).sum()),worsened=int((after>before).sum()))

def macro(rows):
    groups={}
    for q,m in rows:
        groups.setdefault(q['scene'],[]).append(m)
    return float(np.mean([sum(m['MAE_cm']*m['pixels'] for m in group)/sum(m['pixels'] for m in group)
                          for group in groups.values()]))

thresholds={}
for name in names:
    pool=np.concatenate([s[a!=0] for q in val for a,s in [q['proposed'][name]]])
    ts=[float('inf')]+[float(np.quantile(pool,1-f)) for f in (.01,.03,.05,.10,.15,.20,.30,.50) if len(pool)]+[0.]
    readings=[]
    for threshold in ts:
        pairs=[(q,metric(q,apply(*q['proposed'][name],threshold))) for q in val]
        readings.append((macro(pairs),sum(m['edited'] for _,m in pairs),threshold))
    value,edits,th=min(readings)
    thresholds[name]=dict(threshold=th if np.isfinite(th) else None,val_macro_mae_cm=value,val_edits=edits)

best_single=min(names[:3],key=lambda name:thresholds[name]['val_macro_mae_cm'])
best_ensemble=min(names[3:],key=lambda name:thresholds[name]['val_macro_mae_cm'])

def aggregate(rows):
    total=sum(m['pixels'] for m in rows)
    return dict(pixels=total,MAE_cm=sum(m['MAE_cm']*m['pixels'] for m in rows)/total,
                **{k:sum(m[k] for m in rows) for k in ('edited','improved','worsened')})

groups={}
matched={}
shared_budget={}
for grouping in sorted({(q['split'],q['domain']) for q in evaluation.values()}):
    split,domain=grouping
    rows=[q for q in evaluation.values() if (q['split'],q['domain'])==grouping]
    label=f'{split}/{domain}'
    actions={'raw':[np.zeros(len(q['svm']),dtype=int) for q in rows],'P20':[q['svm'] for q in rows]}
    for name in names:
        th=thresholds[name]['threshold']
        actions[name]=[apply(*q['proposed'][name],float('inf') if th is None else th) for q in rows]
    groups[label]={name:aggregate([metric(q,a) for q,a in zip(rows,aa)]) for name,aa in actions.items()}
    matched[label]={}
    # The frozen policies determine the candidate pools. Match their smaller
    # edit budget per clip. P20 is prioritized by lower raw confidence.
    paired=[]
    for q,left,right in zip(rows,actions['P20'],actions[best_ensemble]):
        lidx,ridx=np.flatnonzero(left),np.flatnonzero(right)
        n=min(len(lidx),len(ridx))
        left_keep=lidx[np.argsort(-q['svm_priority'][lidx],kind='stable')[:n]]
        right_keep=ridx[np.argsort(-q['proposed'][best_ensemble][1][ridx],kind='stable')[:n]]
        a,b=np.zeros_like(left),np.zeros_like(right)
        a[left_keep],b[right_keep]=left[left_keep],right[right_keep]
        paired.append((metric(q,a),metric(q,b)))
    shared_budget[label]={'P20-confidence-priority':aggregate([a for a,b in paired]),
                          best_ensemble:aggregate([b for a,b in paired])}

    for name in names:
        per_clip=[]
        unavailable=[]
        for q in rows:
            a,s=q['proposed'][name]
            n=int((q['svm']!=0).sum())
            eligible=np.flatnonzero(a!=0)
            if len(eligible)<n:
                unavailable.append(q['scene'])
                continue
            take=eligible[np.argsort(-s[eligible],kind='stable')[:n]]
            action=np.zeros_like(a)
            action[take]=a[take]
            per_clip.append((metric(q,q['svm']),metric(q,action)))
        matched[label][name]=dict(clips=len(per_clip),total_clips=len(rows),unavailable=unavailable,
             P20=aggregate([m[0] for m in per_clip]) if per_clip else None,
             model=aggregate([m[1] for m in per_clip]) if per_clip else None)
p20.dump(args.out/'summary.json',dict(thresholds=thresholds,best_single=best_single,best_ensemble=best_ensemble,
    validation_raw_macro=macro([(q,metric(q,np.zeros(len(q['svm']),dtype=int))) for q in val]),
    validation_svm_macro=macro([(q,metric(q,q['svm'])) for q in val]),
    train_rows=len(x),train_per_class=ceiling,groups=groups,matched=matched,shared_budget=shared_budget,seconds=time.monotonic()-start))
print('selected',best_single,best_ensemble,flush=True)
