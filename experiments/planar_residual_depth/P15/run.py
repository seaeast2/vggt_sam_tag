"""Merge observed planes across views and project eligible P14 pixels."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'results/planar_residual_depth'
P12 = BASE/'P12-indoor-20261118-001'
P13 = BASE/'P13-indoor-20261118-002'
P14 = BASE/'P14-indoor-20261118-002'
OUT = BASE/'P15-indoor-20261118-001'
OUT.mkdir()
dtype = np.dtype([(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')]
                 +[('confidence','<f4')]+[(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])

def read(path):
    with path.open('rb') as stream:
        header=[]
        for line in stream:
            header.append(line)
            if line.strip()==b'end_header':
                break
        return b''.join(header),np.fromfile(stream,dtype)

def save_json(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

_,source = read(P13/'p13_corrected_rgb.ply')
prev = np.load(P13/'depths.npz')
background = np.load(P14/'depths.npz')
raw = prev['raw']
p14 = background['corrected']
target = background['target']
assert np.array_equal(background['raw'],prev['corrected'])
n,h,w=raw.shape
yy,xx=np.indices((h,w))
holdout=((xx//8+3*(yy//8))%5)==0
assert np.array_equal(source['frame_id'],np.repeat(np.arange(n),h*w))
assert np.array_equal(source['pixel_u'],np.tile(xx.ravel(),n))
assert np.array_equal(source['pixel_v'],np.tile(yy.ravel(),n))
source_xyz=np.stack([source[k] for k in 'xyz'],-1).reshape(n,h,w,3)
centers=[]; matrices=[]; rays=[]
for j in range(n):
    z=prev['corrected'][j].ravel()
    design=np.c_[np.ones(z.size),z,z*xx.ravel(),z*yy.ravel()]
    points=source_xyz[j].reshape(-1,3)
    coef,_,rank,_=np.linalg.lstsq(design[::17],points[::17],rcond=None)
    assert rank==4 and np.max(abs(design@coef-points))<2e-5
    centers.append(coef[0])
    matrices.append(np.linalg.inv(coef[1:]))
    rays.append(np.stack([np.ones_like(xx),xx,yy],-1)@coef[1:])
rays=np.array(rays)
centers=np.array(centers)
raw_xyz=centers[:,None,None,:]+rays*raw[...,None]

def fit_plane(points):
    use=points
    for _ in range(4):
        center=use.mean(0)
        _,s,v=np.linalg.svd(use-center,full_matrices=False)
        normal=v[-1]
        err=abs((points-center)@normal)
        use=points[err<=max(np.quantile(err,.8),1e-7)]
    return normal,float(normal@center),float(s[1]/np.sqrt(len(use)))

regions=np.load(P12/'full.npz')['mask_ids']
nodes=[]
for entry in json.loads((P12/'full.json').read_text())['regions']:
    j,k=entry['frame'],entry['mask']
    region=regions[j]==k
    # Use confidence above the P14 threshold and freeze spatial blocks for audit.
    support=region&~target[j]&~holdout
    if support.sum()<120 or not (region&target[j]).any():
        continue
    y,x=np.where(support)
    if len(np.unique(np.c_[x//16,y//16],axis=0))<6:
        continue
    xyz=raw_xyz[j][support]
    normal,d,spread=fit_plane(xyz[::max(1,len(xyz)//2500)])
    z=float(np.median(raw[j][support]))
    if spread<.025*z or np.quantile(abs(xyz@normal-d),.75)>.025*z:
        continue
    nodes.append(dict(frame=j,mask=k,xyz=xyz,normal=normal,d=d,z=z,support=support))

def overlap(a,b):
    points=a['xyz'][::max(1,len(a['xyz'])//1000)]
    j=b['frame']
    q=(points-centers[j])@matrices[j]
    q=q[q[:,0]>0]
    uv=np.rint(q[:,1:]/q[:,:1]).astype(int)
    x,y=uv.T
    inside=(x>=0)&(x<w)&(y>=0)&(y<h)
    x,y,z=x[inside],y[inside],q[inside,0]
    good=b['support'][y,x]&(abs(z-raw[j,y,x])<=.04*raw[j,y,x])
    pixels=np.unique(np.c_[x[good],y[good]],axis=0)
    return len(pixels)

edges={}
for i,j in combinations(range(len(nodes)),2):
    a,b=nodes[i],nodes[j]
    if a['frame']==b['frame']:
        continue
    dot=float(a['normal']@b['normal'])
    if abs(dot)<np.cos(np.deg2rad(8)):
        continue
    if abs(a['d']-np.sign(dot)*b['d'])>.04*min(a['z'],b['z']):
        continue
    ab,ba=overlap(a,b),overlap(b,a)
    if min(ab,ba)>=30:
        edges[i,j]=min(ab,ba)

groups=[{i} for i in range(len(nodes))]
for i,j in sorted(edges,key=edges.get,reverse=True):
    a=next(g for g in groups if i in g)
    b=next(g for g in groups if j in g)
    if a is not b and all(tuple(sorted((x,y))) in edges for x in a for y in b):
        a.update(b)
        groups.remove(b)

corrected=p14.copy()
assigned=np.full(raw.shape,-1,dtype=np.int16)
records=[]
for group in groups:
    if len(group)<2:
        continue
    members=[nodes[i] for i in sorted(group)]
    samples=np.concatenate([a['xyz'][np.linspace(0,len(a['xyz'])-1, min(512,len(a['xyz'])),dtype=int)] for a in members])
    normal,d,_=fit_plane(samples)
    if any(np.quantile(abs(a['xyz']@normal-d),.75)>.03*a['z'] for a in members):
        continue
    record=dict(normal=normal.tolist(),offset=d,members=[],heldout_plane_distance=[])
    for a in members:
        j,k=a['frame'],a['mask']
        region=regions[j]==k
        den=rays[j]@normal
        plane=np.divide(d-centers[j]@normal,den,out=np.zeros_like(den),where=abs(den)>1e-6)
        # Guard the extrapolation relative to the pre-P14 observation, rather than its synthetic far depth.
        use=region&target[j]&(plane>0)&(abs(plane-raw[j])<=.30*raw[j])
        corrected[j][use]=plane[use]
        assigned[j][use]=len(records)
        test=use&holdout
        if test.any():
            record['heldout_plane_distance'].append(dict(frame=j,pixels=int(test.sum()),
                raw=float(np.median(abs(raw_xyz[j][test]@normal-d))),
                p14=float(np.median(abs((centers[j]+rays[j][test]*p14[j][test,None])@normal-d)))))
        record['members'].append(dict(frame=j,mask=k,applied=int(use.sum())))
    records.append(record)
selected=assigned>=0
assert np.array_equal(corrected[~selected],p14[~selected])
assert (assigned[target]<len(records)).all() and not (selected&~target).any()
np.savez_compressed(OUT/'depths.npz',raw=raw,p14=p14,corrected=corrected,residual_from_p14=corrected-p14,
                    selected=selected,plane_id=assigned)
save_json('planes.json',dict(candidates=len(nodes),pairwise_matches=len(edges),planes=records,
    changed=int(selected.sum()),target=int(target.sum()),
    gates=dict(max_normal_degrees=8,min_bidirectional_pixels=30,depth_agreement=.04,max_raw_depth_change=.30)))

for mode in ['confidence','rgb']:
    header,points=read(P14/f'p14_background_{mode}.ply')
    for k in ['frame_id','pixel_u','pixel_v','confidence']:
        assert np.array_equal(points[k],source[k])
    keep=points[~selected.ravel()].copy()
    xyz=centers[:,None,None,:]+rays*corrected[...,None]
    for k,axis in enumerate('xyz'):
        points[axis][selected.ravel()]=xyz[...,k][selected]
    assert np.array_equal(keep,points[~selected.ravel()])
    path=OUT/f'p15_shared_planes_{mode}.ply'
    # Preserve the point schema and replace stale P13/P14 comments.
    hdr=b''.join(line for line in header.splitlines(keepends=True) if not line.startswith(b'comment'))
    with path.open('wb') as stream:
        stream.write(hdr)
        points.tofile(stream)
    assert np.array_equal(read(path)[1],points)
print('nodes',len(nodes),'edges',len(edges),'groups',len(records),'selected',int(selected.sum()))
