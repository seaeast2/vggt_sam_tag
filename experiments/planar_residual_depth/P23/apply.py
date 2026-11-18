"""Apply a frozen P20 or P22 policy to held-out KITTI frames."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

def import_file(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

scripts=Path(__file__).resolve().parents[1]
p20=import_file('p20',scripts/'P20/run.py')
p21=import_file('p21',scripts/'P21/compare.py')
p=argparse.ArgumentParser()
p.add_argument('dataset',type=Path)
p.add_argument('policy',type=Path)
p.add_argument('out',type=Path)
p.add_argument('--method',choices=['rf-gbdt','svm'],default='rf-gbdt')
args=p.parse_args()
args.out.mkdir(parents=True)
clips=p20.load_manifest(args.dataset)
assert len(clips)==1
clip=clips[0]
assert clip['domain']=='outdoor' and clip['split']=='test'
spec=json.loads((args.policy/'protocol.json').read_text())
assert spec['features']==p20.FEATURES
if args.method=='svm':
    model_path=args.policy/'svm.xml'
    svm=cv2.ml.SVM_load(str(model_path))
    with np.load(args.policy/'normalization.npz',allow_pickle=False) as norms:
        mean,std=norms['mean'],norms['std']
    assert svm.isTrained() and svm.getVarCount()==len(p20.FEATURES)
    assert mean.shape==std.shape==(len(p20.FEATURES),) and (std>0).all()
    threshold=None

    def choose(features):
        normalized=np.ascontiguousarray((features-mean)/std,dtype=np.float32)
        return svm.predict(normalized)[1].ravel().astype(int)
else:
    import joblib
    summary=json.loads((args.policy/'summary.json').read_text())
    assert summary['best_ensemble']=='RF+GBDT'
    threshold=summary['thresholds']['RF+GBDT']['threshold']
    assert threshold is not None
    model_path=args.policy/'models.joblib'
    models=joblib.load(model_path)
    rf=models['classifiers']['RF']

    def choose(features):
        gains=np.column_stack([m.predict(features) for m in models['regressors']])
        direction=gains.argmax(1)+1
        return np.where((rf.predict(features)==direction)&(gains.max(1)>threshold),direction,0)

with np.load(clip['cache'],allow_pickle=False) as source:
    keys=('rgb','raw','confidence','sam','model_K','model_pose')
    cache={k:source[k] for k in keys}
raw=cache['raw']
confidence=cache['confidence']
protect=float(np.quantile(confidence,.70))
normalizer=float(np.median(raw))
corrected=raw.copy()
action=np.zeros_like(raw,dtype=np.uint8)
candidate=np.zeros_like(raw,dtype=bool)
for i,z in enumerate(raw):
    y,x,f,depths=p20.sides(cache['rgb'][i],z,confidence[i],cache['sam'][i],
                          cache['model_K'][i],protect,normalizer)
    candidate[i,y,x]=True
    if not len(f):
        continue
    choices=choose(f)
    action[i,y,x]=choices
    corrected[i,y,x]=depths[np.arange(len(choices)),choices]
assert not ((action!=0)&~candidate).any()
assert not ((action!=0)&(confidence>=protect)).any()
assert np.array_equal(corrected[action==0],raw[action==0])
assert np.isfinite(corrected).all() and (corrected>0).all()
np.savez_compressed(args.out/'depths.npz',raw=raw,corrected=corrected,residual=corrected-raw,
                    action=action,candidate=candidate)
# Export the raw and corrected clouds through the same projection and PLY schema.
for name,depth in [('raw',raw),('corrected',corrected)]:
    folder=args.out/name
    folder.mkdir()
    p21.export(folder,cache,depth)
marked=dict(cache,rgb=cache['rgb'].copy())
changed=corrected!=raw
marked['rgb'][changed]=[255,145,20]
folder=args.out/'actions'
folder.mkdir()
p21.export(folder,marked,corrected)
# Sensor depth is opened only after the actions, depths and PLYs are saved.
with np.load(clip['cache'],allow_pickle=False) as source:
    gt=source['gt']
scale,test=p20.calibrated(gt,raw)
before,after=[abs(z*scale-gt) for z in (raw,corrected)]
def metrics(mask):
    return dict(pixels=int(mask.sum()),raw_MAE_cm=float(100*before[mask].mean()) if mask.any() else None,
                corrected_MAE_cm=float(100*after[mask].mean()) if mask.any() else None,
                improved=int((after[mask]<before[mask]).sum()),worsened=int((after[mask]>before[mask]).sum()))
with model_path.open('rb') as stream:
    digest=hashlib.file_digest(stream,'sha256').hexdigest()
record=dict(status='COMPLETED',method=args.method,policy=str(args.policy),sha256=digest,threshold=threshold,
            clip=clip,frames=len(raw),points=raw.size,confidence_p70=protect,candidate_pixels=int(candidate.sum()),
            changed_depth_pixels=int(changed.sum()), marking=dict(rgb=[255,145,20],mask='corrected_depth != raw_depth'),
            candidate_actions=p20.counts(action[candidate]),gt_scale_for_evaluation=scale,
            evaluation={'all':metrics(test),'candidate':metrics(test&candidate),'modified':metrics(test&(action!=0))})
p20.dump(args.out/'manifest.json',record)
print(json.dumps(record,indent=2))
