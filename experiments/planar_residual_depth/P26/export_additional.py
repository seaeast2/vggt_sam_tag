"""Export a depth-only mask from a raster PLY using its recovered projection."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('folder',type=Path)
p.add_argument('--percent',type=int,choices=[3,5],default=5)
args=p.parse_args()
folder=args.folder
meta=folder/'manifest.json'
record=json.loads(meta.read_text())
n=record['frames']
h,w=record['raster']
dtype=np.dtype([(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')]+
               [('confidence','<f4')]+[(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])
with (folder/'raw.ply').open('rb') as f:
    lines=[]
    for line in f:
        lines.append(line)
        if line==b'end_header\n':
            break
    points=np.fromfile(f,dtype)
assert len(points)==n*h*w
old=np.load(folder/'filter_masks.npz')['edge']
keep=[]
disagreements=[]
max_reprojection=[]
for i,block in enumerate(points.reshape(n,-1)):
    assert (block['frame_id']==i).all()
    xyz=np.stack([block[k] for k in 'xyz'],axis=1).astype(float)
    xy=np.stack([block[k] for k in ('pixel_u','pixel_v')],axis=1).astype(float)
    center,spread=xyz.mean(0),xyz.std(0).max()
    world=np.c_[(xyz-center)/spread,np.ones(len(xyz))]
    uv=(xy-[w/2,h/2])/w
    sample=slice(None,None,37)
    terms=world[sample]
    zero=np.zeros_like(terms)
    mat=np.concatenate([np.c_[terms,zero,-uv[sample,0,None]*terms],
                        np.c_[zero,terms,-uv[sample,1,None]*terms]])
    _,_,v=np.linalg.svd(mat,full_matrices=False)
    camera=v[-1].reshape(3,4)
    projected=world@camera.T
    error=np.max(abs(projected[:,:2]/projected[:,2,None]-uv))*w
    assert error<.01,(i,error)
    max_reprojection.append(float(error))
    z=projected[:,2]
    z=(z/np.median(z)).reshape(h,w).astype('float32')
    assert (z>0).all()
    kernel=np.ones((3,3),dtype=np.uint8)
    jump=(cv2.dilate(z,kernel,borderType=cv2.BORDER_REPLICATE)-
          cv2.erode(z,kernel,borderType=cv2.BORDER_REPLICATE))/z
    disagreement=int(np.count_nonzero((jump>.03)!=old[i]))
    assert disagreement<=10,(i,disagreement)
    disagreements.append(disagreement)
    keep.append(~old[i] if args.percent==3 else jump<=.05)
mask=np.stack(keep)
selected=points[mask.ravel()]
header=b''.join(f'element vertex {len(selected)}\n'.encode() if line.startswith(b'element vertex ') else line for line in lines)
name=f'edge_3x3_{args.percent}pct'
out=folder/f'{name}.ply'
with out.open('xb') as f:
    f.write(header)
    selected.tofile(f)
with out.open('rb') as f:
    assert f.read(len(header))==header and np.array_equal(np.fromfile(f,dtype),selected)
np.savez_compressed(folder/f'{name}_mask.npz', keep=mask)
record.setdefault('additional_filters',{})[out.name]=dict(rule=f'3x3 relative depth range <= {args.percent/100:.2f}; no confidence filtering',
    kept=int(mask.sum()),removed=int((~mask).sum()),depth_source=('original saved 3% depth-edge mask' if args.percent==3 else
                  'per-frame camera Z recovered from raw PLY XYZ-to-pixel projective fit; multiplicative scale cancels'),
    max_reprojection_px=max(max_reprojection),validation_original_3pct_mask_disagreements=sum(disagreements))
meta.write_text(json.dumps(record,indent=2)+'\n')
print(record['additional_filters'][out.name])
