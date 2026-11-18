"""PCA visualization and scene-held-out SVM depth-error detection."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

manifest=Path(sys.argv[1])
out=Path(sys.argv[2])
out.mkdir(parents=True)
rng=np.random.default_rng(19)
splits={key:[] for key in ('train','val','test')}
for row in json.loads(manifest.read_text())['clips']:
    with np.load(manifest.parent/row['cache']) as z:
        depth,gt,conf,sam=[z[k] for k in ('raw','gt','confidence','sam')]
        y,x=np.indices(depth.shape[1:])
        cal=(x//16+3*(y//16))%5==1
        valid=np.isfinite(gt)&(gt>0)
        scale=np.median(gt[valid&cal]/depth[valid&cal])
        normalized=depth/np.median(depth)
        for i,rgb in enumerate(z['rgb']):
            lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            def gradient(a):
                return cv2.magnitude(cv2.Sobel(a,cv2.CV_32F,1,0,ksize=3),
                                     cv2.Sobel(a,cv2.CV_32F,0,1,ksize=3))
            f=np.dstack([lab,np.log(normalized[i]),np.log1p(conf[i]),sam[i],
                         gradient(gray),gradient(np.log(normalized[i]).astype('float32'))])
            ids=np.flatnonzero(valid[i]&~cal)
            ids=rng.choice(ids,min(len(ids),1000),replace=False)
            error=np.abs(depth[i].ravel()[ids]*scale-gt[i].ravel()[ids])/gt[i].ravel()[ids]
            splits[row['split']].append((f.reshape(-1,8)[ids],error>.10,row['scene'],row['domain']))

def unpack(rows):
    return np.concatenate([r[0] for r in rows]),np.concatenate([r[1] for r in rows])

x,y=unpack(splits['train'])
mean,std=x.mean(0),x.std(0).clip(1e-6)
x=(x-mean)/std
_,s,v=np.linalg.svd(x,full_matrices=False)
variance=s*s/(s*s).sum()
k=int(np.searchsorted(np.cumsum(variance),.95))+1
features={'all':lambda a:a,'pca95':lambda a:a@v[:k].T}
# Equal class sizes, fixed before using validation or test.
count=min(2500,int(y.sum()),int((~y).sum()))
ids=np.r_[rng.choice(np.flatnonzero(y),count,replace=False),
          rng.choice(np.flatnonzero(~y),count,replace=False)]
models={}
for name,transform in features.items():
    train=transform(x[ids]).astype('float32')
    svm=cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(2)
    svm.setGamma(1/train.shape[1])
    assert svm.train(train,cv2.ml.ROW_SAMPLE,y[ids].astype('int32'))
    score=svm.predict(train,flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1].ravel()
    models[name]=(svm,float(np.sign(score[y[ids]].mean()-score[~y[ids]].mean())))
    svm.save(str(out/f'svm_{name}.xml'))

def auc(y,s):
    n=int(y.sum()); m=len(y)-n
    return float((rankdata(s)[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None

feature_names=['Lab-L','Lab-a','Lab-b','log_depth','log1p_conf','SAM_distance','RGB_gradient','depth_gradient']
report=dict(features=feature_names,pca_components_95=k,
            explained_variance_ratio=variance.tolist(),loadings=v.tolist(),mean=mean.tolist(),std=std.tolist(),
            label='relative depth error >10%; scale calibrated per clip on disjoint GT blocks',
            svm_balanced_train_pixels=2*count,results={})
fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
domains=[('validation',splits['val']),
         ('Middlebury test',[r for r in splits['test'] if r[3]=='indoor-stereo']),
         ('TUM test',[r for r in splits['test'] if r[3]=='indoor'])]
for ax,(name,rows) in zip(axs,domains):
    f,y=unpack(rows)
    standard=(f-mean)/std
    scores={'confidence_only':-f[:,4]}
    for mode,(svm,sign) in models.items():
        scores[mode]=sign*svm.predict(features[mode](standard).astype('float32'),
                                      flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1].ravel()
    report['results'][name]=dict(pixels=len(y),large_error=int(y.sum()),scenes=sorted({r[2] for r in rows}),
                                  auc={key:auc(y,value) for key,value in scores.items()})
    projection=standard@v[:2].T
    for flag,color,label in [(False,'#346d9e','≤10% error'),(True,'#d65439','>10% error')]:
        subset=projection[y==flag]
        ax.scatter(*subset.T,s=4,alpha=.20,color=color,label=label)
    ax.set(title=name,xlabel='PC1',ylabel='PC2')
    ax.legend(markerscale=2)
fig.savefig(out/'pca_depth_error.png',dpi=160)
(out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(pca95=k,pc2_variance=sum(variance[:2]),results=report['results']),indent=2))
