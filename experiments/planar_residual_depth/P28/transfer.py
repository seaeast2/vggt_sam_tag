"""Transfer the frozen P27 wall to earlier views, with a separate offset-only adaptation."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

p=argparse.ArgumentParser()
p.add_argument('geometry',type=Path)
p.add_argument('plane',type=Path)
p.add_argument('out',type=Path)
args=p.parse_args()
out=args.out
parent=json.loads((args.plane/'summary.json').read_text())
normal=np.asarray(parent['normal'])
offset=parent['offset']
prior_ids={int(i) for i in parent['per_frame']}
with np.load(args.geometry/'input.npz') as z:
    raw,xyz,keep,conf,rgb,P=[z[k] for k in ('depth','xyz','keep','confidence','rgb','cameras')]
with np.load(args.geometry/'wall_support.npz') as z:
    old_wall=z['mask']
wall=np.load(out/'sam_mask.npy')
ids=[int(i) for i in np.where(wall.any(axis=(1,2)))[0]]
assert not set(ids)&prior_ids
n,h,w=raw.shape
y,x=np.indices((h,w))
uv=np.stack([x,y,np.ones_like(x)],-1)
centers=np.array([-np.linalg.solve(p[:,:3],p[:,3]) for p in P])
rays=np.array([uv@np.linalg.inv(p[:,:3]).T for p in P])
spatial_test=((x//12+3*(y//12))%5)==0
regions=np.array([cv2.erode(m.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool) for m in wall])&keep
positions={}
for i in ids:
    support=regions[i]&~spatial_test
    use=support&(conf[i]>=np.median(conf[i][support]))
    assert use.sum()>=100
    positions[i]=float(np.median(xyz[i][use]@normal))
anchors=float(np.median(list(positions.values())))
arms={'frozen_plane':offset,'frozen_direction_reanchored':anchors}
proposals={}
common=regions.copy()
for name,d in arms.items():
    denom=rays@normal
    z=np.divide((d-centers@normal)[:,None,None],denom,out=np.zeros_like(denom),where=abs(denom)>1e-8)
    common&=(z>0)&(abs(z-raw)<=.30*raw)
    proposals[name]=z
# This measures whether the target mask projects into any of the original fitted masks
# under the frozen plane. It is a support diagnostic, not correspondence ground truth.
overlap={}
for i in ids:
    points=centers[i]+rays[i][regions[i]]*proposals['frozen_plane'][i][regions[i],None]
    any_seen=np.zeros(len(points),dtype=bool)
    for j in prior_ids:
        image=points@P[j,:,:3].T+P[j,:,3]
        uu,vv=np.rint(image[:,:2]/image[:,2,None]).astype(int).T
        valid=(image[:,2]>0)&(uu>=0)&(uu<w)&(vv>=0)&(vv<h)
        any_seen[valid]|=old_wall[j,vv[valid],uu[valid]]
    overlap[i]=float(any_seen.mean())
dtype=np.dtype([(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')]+
               [('confidence','<f4')]+[(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])
visible=keep&np.isin(np.arange(n),ids)[:,None,None]
def export(path,points,visibility,changed):
    rec=np.empty(visibility.sum(),dtype)
    rec['frame_id'],rec['pixel_v'],rec['pixel_u']=np.where(visibility)
    rgb_selected=rgb[visibility].copy()
    rgb_selected[changed[visibility]]=[255,145,20]
    for j,k in enumerate('xyz'):
        rec[k]=points[...,j][visibility]
    for j,k in enumerate(('red','green','blue')):
        rec[k]=rgb_selected[:,j]
    rec['confidence']=conf[visibility]
    assert np.isfinite(points[visibility]).all()
    header=(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(rec)}\n'
            'property float x\nproperty float y\nproperty float z\nproperty uchar red\n'
            'property uchar green\nproperty uchar blue\nproperty float confidence\n'
            'property ushort frame_id\nproperty ushort pixel_u\nproperty ushort pixel_v\nend_header\n').encode()
    with path.open('xb') as f:
        f.write(header)
        rec.tofile(f)
    with path.open('rb') as f:
        assert f.read(len(header))==header and np.array_equal(np.fromfile(f,dtype),rec)
nochange=np.zeros_like(common)
for subset,label in [(visible,'full'),(visible&wall,'wall')]:
    export(out/f'raw_{label}.ply',xyz,subset,nochange)
for name,z in proposals.items():
    points=xyz.copy()
    f=np.where(common)[0]
    points[common]=centers[f]+z[common,None]*rays[common]
    changed=(points!=xyz).any(-1)
    assert not (changed&~common).any()
    for subset,label in [(visible,'full'),(visible&wall,'wall')]:
        export(out/f'{name}_{label}.ply',points,subset,changed)
    proposals[name]=np.where(common,z,raw).astype('float32')
np.savez_compressed(out/'depths.npz',raw=raw,selected=common,spatial_test=spatial_test,**proposals)
sheet=Image.new('RGB',(w*len(ids),h+24),'white')
draw=ImageDraw.Draw(sheet)
for index,i in enumerate(ids):
    img=rgb[i].copy()
    img[wall[i]]=np.rint(.55*img[wall[i]]+.45*np.array([90,160,255])).astype('uint8')
    img[common[i]]=[255,145,20]
    sheet.paste(Image.fromarray(img),(index*w,24))
    draw.text((index*w+3,4),f'{i:02} raw anchors and frozen plane | {common[i].sum()} pixels',fill='black')
sheet.save(out/'application.jpg',quality=93)
save=dict(status='COMPLETED_EXPLORATORY',source=str(args.geometry),plane_source=str(args.plane),
          fit_views=sorted(prior_ids),test_views=ids,normal=normal.tolist(),offsets=arms,
          new_view_anchor_offsets=positions,original_mask_overlap_fraction=overlap,
          common_projected_pixels=int(common.sum()),per_frame={i:int(common[i].sum()) for i in ids},
          fixed_filter='edge_3x3_3pct from P26',cap=.30,
          scope='same VGGT 16-frame inference, earlier frames excluded from plane fit; physical wall identity not independently labeled; no GT loaded')
(out/'summary.json').write_text(json.dumps(save,indent=2)+'\n')
print(save)
