"""Evaluate frozen P15 outputs on the same RGB-D pixels in each arm."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

root=Path(__file__).resolve().parents[3]
out=root/'results/planar_residual_depth/P15-indoor-20261118-001'
dataset=root/'data/planar_residual_depth/P10/source/rgbd_dataset_freiburg3_structure_texture_far'
def entries(name):
    return [line.split() for line in (dataset/name).read_text().splitlines() if line and not line.startswith('#')]
rgb,depth=entries('rgb.txt'),entries('depth.txt')
z=np.load(out/'depths.npz')
raw=z['raw']
n,h,w=raw.shape
gt=[]
timestamps=[]
for i in range(n):
    desired=float(rgb[0][0])+2*i
    t=float(min(rgb,key=lambda row:abs(float(row[0])-desired))[0])
    stamp,path=min(depth,key=lambda row:abs(float(row[0])-t))
    assert abs(float(stamp)-t)<.03
    gt.append(np.asarray(Image.open(dataset/path).resize((w,h),Image.Resampling.NEAREST))/5000.)
    timestamps.append(t)
gt=np.stack(gt)
y,x=np.indices((h,w))
cal=((x//16+3*(y//16))%5)==1
held=((x//8+3*(y//8))%5)==0
valid=gt>0
scale=float(np.median(gt[valid&cal]/raw[valid&cal]))
test=valid&~cal&held&z['selected']
def scores(mask):
    return {k:float(100*np.mean(abs(z[k][mask]*scale-gt[mask]))) if mask.any() else None
            for k in ('raw','p14','corrected')}
stats=dict(scale_evaluation_only=scale,timestamps=timestamps,
    projected_pixel_count=int(z['selected'].sum()),gt_test_count=int(test.sum()),
    MAE_cm_on_same_projected_pixels=scores(test),
    per_plane=[dict(plane=i,pixels=int((test&(z['plane_id']==i)).sum()),
                    MAE_cm=scores(test&(z['plane_id']==i))) for i in range(3)])
(out/'evaluation.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats,indent=2))
