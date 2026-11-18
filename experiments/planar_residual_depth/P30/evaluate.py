"""Evaluate identical KITTI bollard pixels with post-fit Velodyne projection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'P10'))
from evaluate import zbuffer

p=argparse.ArgumentParser()
p.add_argument('prediction',type=Path)
p.add_argument('run',type=Path)
p.add_argument('case',type=Path)
a=p.parse_args()
summary=json.loads((a.run/'summary.json').read_text())
case=json.loads(a.case.read_text())
by_path={str(v['source']):v for v in case['images']}
mask=np.load(a.run/'sam_mask.npy')
rgb=np.load(a.prediction)['rgb']
with np.load(a.run/'depths.npz') as data:
    raw,selected=data['raw'],data['selected']
    predictions={k:data[k] for k in ('raw','cylinder','tapered')}
h,w=raw.shape[1:]
gt=[]
provenance=[]
for i,row in enumerate(summary['source_frames']):
    view=by_path[row['source']]
    source=Path(row['source'])
    assert row['sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    with Image.open(source) as image:
        ow,oh=image.size
        cw,ch=min(ow,2*oh),min(oh,2*ow)
        left,top=(ow-cw)//2,(oh-ch)//2
        crop=image.convert('RGB').crop((left,top,left+cw,top+ch))
        assert np.array_equal(np.asarray(crop.resize((w,h),Image.Resampling.BICUBIC)),rgb[i])
    # Undo the preparation raster transform, then map to the exact P25 crop.
    original_k=np.linalg.solve(view['A'],view['K'])
    affine=np.array([[w/cw,0,(-left+.5)*w/cw-.5],
                     [0,h/ch,(-top+.5)*h/ch-.5],[0,0,1]])
    scan=Path(view['GT_velodyne_path'])
    points=np.fromfile(scan,'float32').reshape(-1,4)[:,:3]
    transform=np.asarray(case['metadata']['T_camera2_velodyne'])
    points=points@transform[:3,:3].T+transform[:3,3]
    depth,_=zbuffer(points,affine@original_k,(h,w))
    gt.append(depth)
    provenance.append(dict(scan=str(scan),sha256=hashlib.sha256(scan.read_bytes()).hexdigest()))
gt=np.stack(gt)
valid=np.isfinite(gt)&(gt>0)
y,x=np.indices((h,w))
cal=valid&~mask&(((x//16+3*y//16)%5)==1)
scale=float(np.median(gt[cal]/raw[cal]))
errors={name:abs(z*scale-gt) for name,z in predictions.items()}

def metrics(region):
    use=valid&region
    return dict(count=int(use.sum()),MAE_m={k:float(v[use].mean()) if use.any() else None
                                           for k,v in errors.items()})

result=dict(heldout=metrics(selected&np.isin(np.arange(len(raw)),summary['heldout_frames'])[:,None,None]),
            all_changed_candidates=metrics(selected),all_mask=metrics(mask),
            per_frame={i:metrics(selected& (np.arange(len(raw))==i)[:,None,None]) for i in range(len(raw))},
            scale=scale,scale_support_pixels=int(cal.sum()),scan_provenance=provenance,
            scope='minimum camera-Z per raster pixel; no compensation for scan motion or manual verification of object ownership')
(a.run/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
