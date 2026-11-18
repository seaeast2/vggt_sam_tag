"""Assign all red-to-orange source pixels a background depth."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

base = Path(__file__).resolve().parents[3]/'results/planar_residual_depth'
parent = base/'P13-indoor-20261118-002'
out = base/'P14-indoor-20261118-002'
out.mkdir()
dtype=np.dtype([(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')]+[('confidence','<f4')]+[(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])
def read(path):
    with path.open('rb') as stream:
        header=[]
        for line in stream:
            header.append(line)
            if line.strip()==b'end_header':
                break
        return b''.join(header),np.fromfile(stream,dtype)

header, cloud = read(parent/'p13_corrected_confidence.ply')
_, rgb = read(parent/'p13_corrected_rgb.ply')
data = np.load(parent/'depths.npz')
raw = data['corrected']
f,y,x = [cloud[k] for k in ('frame_id','pixel_v','pixel_u')]
assert len(cloud)==raw.size and np.array_equal(f.astype(np.int64)*raw.shape[1]*raw.shape[2]+y.astype(np.int64)*raw.shape[2]+x,np.arange(raw.size))
confidence = cloud['confidence'].reshape(raw.shape)
low_red,high_blue=1.1071243071556092,33.4366423034668
# Midpoint of the red-to-yellow segment in the P10 confidence palette.
threshold = low_red + .125*(high_blue-low_red)
target = confidence <= threshold
corrected = raw.copy()
rows = []
for i, (depth, conf, low) in enumerate(zip(raw, confidence, target)):
    points = np.argwhere(low)
    ref = np.argwhere((conf > threshold)&np.isfinite(depth)&(depth>0))
    assert len(ref)>0
    dist, index = cKDTree(ref).query(points,k=min(128,len(ref)))
    candidates = depth[tuple(ref[index].transpose(2,0,1))]
    initial = depth[low]
    eligible = (dist <= 32)&(candidates>initial[:,None])
    covered = eligible.any(1)
    nearest = eligible.argmax(1)
    result = np.maximum(initial,float(depth.max()))
    result[covered] = candidates[np.arange(len(points)),nearest][covered]
    corrected[i][low] = result
    rows.append(dict(frame=i,target=len(points),local=int(covered.sum()),fallback=int((~covered).sum()),
                     unchanged=int((result==initial).sum())))
changed = corrected != raw
assert np.all(corrected[target]>=raw[target]) and not (changed&~target).any()
np.savez_compressed(out/'depths.npz',raw=raw,corrected=corrected,residual=corrected-raw,selected=changed,target=target)
# Recover the affine unprojection directly from P13's complete raster PLY.
xyz = np.stack([cloud[k] for k in 'xyz'],axis=1).astype(float)
for i in range(len(raw)):
    frame = f==i
    z = raw[i].ravel().astype(float)
    u,v = x[frame].astype(float),y[frame].astype(float)
    design = np.column_stack((np.ones(len(z)),z,z*u,z*v))
    coef,_,rank,_ = np.linalg.lstsq(design[::17],xyz[frame][::17],rcond=None)
    assert rank==4
    reprojection_error = np.max(abs(design@coef-xyz[frame]))
    assert reprojection_error < 2e-5,(i,reprojection_error)
    move = frame & changed.ravel()
    ratio = corrected.ravel()[move]/raw.ravel()[move]
    xyz[move] = coef[0]+(xyz[move]-coef[0])*ratio[:,None]
for k,axis in enumerate('xyz'):
    cloud[axis][changed.ravel()] = xyz[changed.ravel(),k]
for name,mode in [('p14_background_confidence.ply','confidence'),('p14_background_rgb.ply','rgb')]:
    if mode=='rgb':
        for key in ('red','green','blue'):
            cloud[key]=rgb[key]
    path=out/name
    with path.open('wb') as stream:
        stream.write(header)
        cloud.tofile(stream)
    assert np.array_equal(read(path)[1],cloud)
summary=dict(experiment='P14',parent=parent.name,threshold=float(threshold),palette_fraction=.125,
    local_radius_pixels=32,max_nearest_candidates=128,
    fallback='maximum depth in same frame, clamped to at least input depth',
    frames=rows,total={key:sum(r[key] for r in rows) for key in ['target','local','fallback','unchanged']},
    changed=int(changed.sum()))
(out/'protocol.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary['total'], 'changed',summary['changed'],'threshold',threshold)
