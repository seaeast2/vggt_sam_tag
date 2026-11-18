"""Prepare P26 raster RGB and recover camera projection from saved provenance."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw

source=Path(sys.argv[1])
out=Path(sys.argv[2])
out.mkdir(parents=True)
metadata=json.loads((source/'manifest.json').read_text())
n=metadata['frames']
h,w=metadata['raster']
dtype=np.dtype([(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')]+
               [('confidence','<f4')]+[(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])
with (source/'raw.ply').open('rb') as stream:
    for line in stream:
        if line.strip()==b'end_header':break
    records=np.fromfile(stream,dtype)
assert len(records)==n*h*w
y,x=np.indices((h,w))
rgb=np.stack([records[k] for k in ('red','green','blue')],-1).reshape(n,h,w,3)
xyz=np.stack([records[k] for k in 'xyz'],-1).reshape(n,h,w,3)
confidence=records['confidence'].reshape(n,h,w)
keep=np.load(source/'edge_3x3_3pct_mask.npz')['keep']
assert keep.shape==(n,h,w)
cams=[]
errors=[]
depth=[]
for i,block in enumerate(records.reshape(n,-1)):
    assert (block['frame_id']==i).all()
    assert np.array_equal(block['pixel_u'],x.ravel()) and np.array_equal(block['pixel_v'],y.ravel())
    points=xyz[i].reshape(-1,3).astype(float)
    center=points.mean(0)
    spread=points.std(0).max()
    world=np.c_[(points-center)/spread,np.ones(len(points))]
    uv=(np.c_[x.ravel(),y.ravel()]-[w/2,h/2])/w
    s=world[::37]
    zero=np.zeros_like(s)
    a=np.concatenate([np.c_[s,zero,-uv[::37,0,None]*s],np.c_[zero,s,-uv[::37,1,None]*s]])
    _,_,vh=np.linalg.svd(a,full_matrices=False)
    p=vh[-1].reshape(3,4)
    transform=np.eye(4)
    transform[:3,:3]/=spread
    transform[:3,3]=-center/spread
    p=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]])@p@transform
    p/=np.linalg.norm(p[2,:3])
    proj=np.c_[points,np.ones(len(points))]@p.T
    if np.median(proj[:,2])<0:
        proj=-proj
        p=-p
    error=np.max(abs(proj[:,:2]/proj[:,2,None]-np.c_[x.ravel(),y.ravel()]))
    assert error<.01 and (proj[:,2]>0).all()
    errors.append(float(error))
    cams.append(p)
    depth.append(proj[:,2].reshape(h,w))
np.savez_compressed(out/'input.npz',rgb=rgb,xyz=xyz,confidence=confidence,keep=keep,
                    cameras=np.array(cams),depth=np.array(depth,dtype='float32'))
(out/'input.json').write_text(json.dumps(dict(source=str(source),max_reprojection_px=max(errors),
                      frames=n,filter='edge_3x3_3pct; frozen P26 mask'),indent=2)+'\n')
sheet=Image.new('RGB',(4*w,4*(h+24)),'white')
draw=ImageDraw.Draw(sheet)
for i,im in enumerate(rgb):
    x,y=i%4*w,i//4*(h+24)
    sheet.paste(Image.fromarray(im),(x,y+24))
    draw.text((x+5,y+5),f'Frame {i:02}',fill='black')
sheet.save(out/'source_frames.jpg',quality=94)
