"""Evaluate the frozen P28 outputs on matched RGB-D pixels."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
p.add_argument('dataset',type=Path)
p.add_argument('--compare',type=Path,help='Compare a second run on the intersection of applied pixels')
args=p.parse_args()
out=args.run
s=json.loads((out/'summary.json').read_text())
with np.load(Path(s['source'])/'input.npz') as z:
    rgb,cam=z['rgb'],z['cameras']
with np.load(out/'depths.npz') as z:
    raw,selected,hold=[z[k] for k in ('raw','selected','spatial_test')]
    arms={k:z[k] for k in s['offsets']}
mask=np.load(out/'sam_mask.npy')
def rows(name):
    return [line.split() for line in (args.dataset/name).read_text().splitlines()
            if line and not line.startswith('#')]
images,depths=rows('rgb.txt'),rows('depth.txt')
h,w=raw.shape[1:]
gt=[]
for i,image in enumerate(rgb):
    time=float(images[0][0])+2*i
    t,p=min(images,key=lambda row:abs(float(row[0])-time))
    a=np.asarray(Image.open(args.dataset/p).convert('RGB').resize((w,h),Image.Resampling.BICUBIC))
    assert np.array_equal(a,image),i
    stamp,p=min(depths,key=lambda row:abs(float(row[0])-float(t)))
    assert abs(float(stamp)-float(t))<=.03
    gt.append(np.asarray(Image.open(args.dataset/p).resize((w,h),Image.Resampling.NEAREST))/5000.)
gt=np.stack(gt)
y,x=np.indices((h,w))
cal=(x//16+3*(y//16))%5==1
valid=(gt>0)&np.isfinite(gt)
scale=float(np.median(gt[valid&cal]/raw[valid&cal]))
base=valid&~cal&selected
test=base&hold
errors={k:abs(z*scale-gt) for k,z in [('raw',raw),*arms.items()]}
def metrics(selection,values):
    return dict(pixels=int(selection.sum()),MAE_cm={k:float(100*v[selection].mean()) if selection.any() else None
                                                   for k,v in values.items()})
K=np.array([[535.4*w/640,0,(320.1+.5)*w/640-.5],[0,539.2*h/480,(247.6+.5)*h/480-.5],[0,0,1]])
rays=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(K).T
result=dict(scope='same changed pixels, spatial blocks excluded from reanchor fit, scale fitted on separate sensor pixels',
            scale=scale,primary=metrics(test,errors),all_selected=metrics(base,errors),frames={})
if args.compare:
    comparison=json.loads((args.compare/'summary.json').read_text())
    if comparison['source']!=s['source'] or comparison['test_views']!=s['test_views']:
        raise ValueError('Compared runs must share geometry and target frames')
    with np.load(args.compare/'depths.npz') as z:
        if not (np.array_equal(raw,z['raw']) and np.array_equal(hold,z['spatial_test'])):
            raise ValueError('Compared runs must share raw depth and spatial split')
        common=test&z['selected']
        comparison_errors={k:abs(z[k]*scale-gt) for k in comparison['offsets']}
    if errors.keys()&comparison_errors.keys():
        raise ValueError('Compared variants must have distinct names')
    result['comparison']={
        'source':str(args.compare),
        'primary_intersection':metrics(common,errors|comparison_errors),
    }
for i in s['test_views']:
    active=mask[i]&valid[i]
    points=rays[active]*gt[i][active,None]
    use=np.ones(len(points),dtype=bool)
    for _ in range(4):
        c=points[use].mean(0)
        _,_,v=np.linalg.svd(points[use]-c,full_matrices=False)
        normal=v[-1]
        e=abs((points-c)@normal)
        use=e<=max(.001,np.quantile(e,.8))
    R=cv2.decomposeProjectionMatrix(cam[i])[1]
    angle=float(np.degrees(np.arccos(np.clip(abs((R@np.array(s['normal']))@normal),0,1))))
    result['frames'][i]=dict(metrics(test[i],{k:v[i] for k,v in errors.items()}),reference_points=len(points),
                             reference_median_residual_cm=float(100*np.median(e)),normal_difference_deg=angle)
(out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
