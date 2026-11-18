"""Audit line-based wall projection against timestamp-matched sensor depth."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
p.add_argument('dataset',type=Path)
p.add_argument('reference',type=Path)
args=p.parse_args()
run=args.run
meta=json.loads((run/'summary.json').read_text())
src=Path(meta['parent'])
with np.load(run/'depths.npz') as a:
    raw,corrected,changed=[a[k] for k in ('raw','corrected','changed')]
with np.load(src/'input.npz') as a:
    rgb,cam=a['rgb'],a['cameras']
audit=json.loads(args.reference.read_text())['frames']
def rows(name):
    return [line.split() for line in (args.dataset/name).read_text().splitlines()
            if line and not line.startswith('#')]
colors,depths=rows('rgb.txt'),rows('depth.txt')
h,w=raw.shape[1:]
gt=[]
for i,image in enumerate(rgb):
    target=float(colors[0][0])+2*i
    t,v=min(colors,key=lambda row:abs(float(row[0])-target))
    assert np.array_equal(image,np.asarray(Image.open(args.dataset/v).convert('RGB').resize((w,h),Image.Resampling.BICUBIC)))
    s,z=min(depths,key=lambda row:abs(float(row[0])-float(t)))
    assert abs(float(s)-float(t))<=.03
    gt.append(np.asarray(Image.open(args.dataset/z).resize((w,h),Image.Resampling.NEAREST))/5000.)
gt=np.stack(gt)
y,x=np.indices((h,w))
cal=(x//16+3*(y//16))%5==1
valid=gt>0
scale=float(np.median(gt[valid&cal]/raw[valid&cal]))
test=valid&~cal&changed
errors=[abs(z*scale-gt) for z in (raw,corrected)]
def mae(mask, arrays=errors):
    return [float(100*e[mask].mean()) if mask.any() else None for e in arrays]
result=dict(test_pixels=int(test.sum()),mae_cm=mae(test),scale_from_raw=scale,per_frame={})
for i in range(12,16):
    R=cv2.decomposeProjectionMatrix(cam[i])[1]
    info=audit[str(i)]
    ref=np.array(info['reference_normal_camera'])
    ang=float(np.degrees(np.arccos(np.clip(abs((R@np.array(meta['normal']))@ref),0,1))))
    result['per_frame'][i]=dict(pixels=int(test[i].sum()),mae_cm=mae(test[i],[e[i] for e in errors]),
        angle_to_sensor_ref_deg=ang,reference_plane_median_residual_m=info['reference_median_residual_m'])
(run/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
