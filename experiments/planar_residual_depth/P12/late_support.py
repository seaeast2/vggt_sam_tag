"""Audit whether later high-confidence observations cover the earlier provisional masks."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[3]
base = root/'results/planar_residual_depth'
out = base/'P12-indoor-20261118-001'
cams = np.load(base/'P10raw-three-sequences-20261118-003/indoor/predictions.npz')
early = json.loads((out/'early.json').read_text())
late = json.loads((out/'full.json').read_text())
depth = cams['depth'][0,...,0]
conf = cams['depth_conf'][0]
K = cams['intrinsics'][0]
ext = cams['extrinsics'][0]
h,w = depth.shape[1:]
clouds = []
for i in range(8,16):
    y,x = np.where(conf[i] >= late['high'])
    camera = (np.c_[x,y,np.ones_like(x)]@np.linalg.inv(K[i]).T)*depth[i,y,x,None]
    clouds.append((camera-ext[i,:,3])@np.linalg.inv(ext[i,:,:3]).T)
world = np.concatenate(clouds)
rows = []
for r in early['regions']:
    if r['mode'] != 'provisional':
        continue
    i,mid = r['frame'],r['mask']
    mask = np.load(base/f'P12-sam-indoor-20261118-001/{i:02}.npz')['masks'][mid]
    camera = world@ext[i,:,:3].T+ext[i,:,3]
    pos = camera[:,2]>0
    camera = camera[pos]
    image = camera@K[i].T
    uv = np.rint(image[:,:2]/image[:,2:]).astype(int)
    x,y = uv.T
    inside = (x>=0)&(x<w)&(y>=0)&(y<h)
    x,y,z = x[inside],y[inside],camera[inside,2]
    # Check geometric agreement and coverage of the same mask in the early view.
    accept = mask[y,x] & (abs(z-depth[i,y,x]) <= .05*depth[i,y,x])
    pixels = np.unique(np.c_[x[accept],y[accept]],axis=0)
    tiles = np.unique(pixels//16,axis=0)
    rows.append(dict(frame=i,mask=mid,support_pixels=len(pixels),support_tiles=len(tiles),
                     coverage=float(len(pixels)/mask.sum()),
                     candidate=bool(len(pixels)>=120 and len(tiles)>=6 and len(pixels)>=.1*mask.sum())))
(out/'late_support_audit.json').write_text(json.dumps(dict(
    policy='later high-confidence points projected with full-context poses; 5% depth agreement, >=120 unique pixels, >=6 tiles, >=10% coverage; exploratory eligibility audit, no replacement',
    regions=rows),indent=2))
print('later coverage candidates',sum(r['candidate'] for r in rows),'/',len(rows))
