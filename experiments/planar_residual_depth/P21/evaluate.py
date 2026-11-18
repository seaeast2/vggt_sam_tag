"""Compare completed P21 clouds using the same frozen GT pixels and scale."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from compare import CACHE

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
args=p.parse_args()
with np.load(CACHE) as z:
    gt,raw=z['gt'],z['raw']
y,x=np.indices(gt.shape[1:])
cal=(x//16+3*(y//16))%5==1
valid=np.isfinite(gt)&(gt>0)
scale=float(np.median(gt[valid&cal]/raw[valid&cal]))
test=valid&~cal
edge=np.zeros_like(valid)
for dim in (1,2):
    left,right=[np.take(gt,indices=range(i,gt.shape[dim]-1+i),axis=dim) for i in (0,1)]
    both=(left>0)&(right>0)&np.isfinite(left)&np.isfinite(right)
    discontinuity=both&(abs(left-right)>.10*np.minimum(left,right))
    if dim==1:
        edge[:,:-1]|=discontinuity
        edge[:,1:]|=discontinuity
    else:
        edge[:,:,:-1]|=discontinuity
        edge[:,:,1:]|=discontinuity
edge=np.array([cv2.dilate(e.astype('uint8'),np.ones((5,5),dtype='uint8'))>0 for e in edge])
result={}
for f in sorted(args.run.glob('*/manifest.json')):
    meta=json.loads(f.read_text())
    if meta.get('status')!='COMPLETED':
        continue
    depth=np.load(f.parent/'depth.npy')
    err=abs(depth*scale-gt)
    result[f.parent.name]=dict(all_MAE_cm=float(100*err[test].mean()),
        GT_edge_band_MAE_cm=float(100*err[test&edge].mean()),
        updated_pixels=int((depth!=raw).sum()))
report=dict(scope='same TUM six frames, comparison in frozen VGGT camera frame',
    gt_scale_from_raw_calibration_blocks=scale,all_test_pixels=int(test.sum()),gt_edge_test_pixels=int((test&edge).sum()),
    metrics=result)
(args.run/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
