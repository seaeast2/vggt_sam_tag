"""Recover the original TUM raster cloud from its P13 provenance and filter visibility."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

p=argparse.ArgumentParser()
p.add_argument('p13',type=Path)
p.add_argument('out',type=Path)
args=p.parse_args()
args.out.mkdir(parents=True)
dtype=np.dtype([(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')]
               +[('confidence','<f4')]+[(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])
with (args.p13/'p13_corrected_rgb.ply').open('rb') as f:
    lines=[]
    for line in f:
        lines.append(line)
        if line==b'end_header\n':
            break
    assert lines[-1]==b'end_header\n'
    rgba=np.fromfile(f,dtype)
with np.load(args.p13/'depths.npz',allow_pickle=False) as z:
    raw,corrected=z['raw'],z['corrected']
n,h,w=raw.shape
y,x=np.indices((h,w))
assert np.array_equal(rgba['frame_id'],np.repeat(np.arange(n),h*w))
assert np.array_equal(rgba['pixel_u'],np.tile(x.ravel(),n))
assert np.array_equal(rgba['pixel_v'],np.tile(y.ravel(),n))
assert np.isfinite(raw).all() and (raw>0).all()
cloud=rgba.copy()
fit_error=[]
for i in range(n):
    block=cloud[i*h*w:(i+1)*h*w]
    xyz=np.stack([block[k] for k in 'xyz'],1).astype(float)
    z=corrected[i].ravel().astype(float)
    design=np.column_stack([np.ones_like(z),z,z*x.ravel(),z*y.ravel()])
    pose,_,rank,_=np.linalg.lstsq(design[::17],xyz[::17],rcond=None)
    error=float(np.max(abs(design@pose-xyz)))
    assert rank==4 and error<2e-5,(i,error)
    fit_error.append(error)
    changed=raw[i].ravel()!=z
    new=pose[0]+(xyz[changed]-pose[0])*(raw[i].ravel()[changed]/z[changed])[:,None]
    for j,k in enumerate('xyz'):
        block[k][changed]=new[:,j]
conf=cloud['confidence'].reshape(raw.shape)
assert np.isfinite(conf).all()
kernel=np.ones((3,3),dtype=np.uint8)
edge=np.array([(cv2.dilate(d,kernel,borderType=cv2.BORDER_REPLICATE)-
                cv2.erode(d,kernel,borderType=cv2.BORDER_REPLICATE))/
               np.maximum(abs(d),1e-6)>.03 for d in raw])
masked_conf=np.where(edge,0,conf)
threshold=float(np.quantile(masked_conf,.2))
keep=(masked_conf>=threshold)&(masked_conf>1e-5)
marked=cloud.copy()
for name,value in zip(('red','green','blue'),(255,145,20)):
    marked[name][~keep.ravel()]=value

def save(name,points):
    header=b''.join((f'element vertex {len(points)}\n'.encode() if l.startswith(b'element vertex ') else l)
                    for l in lines if not l.startswith(b'comment'))
    with (args.out/name).open('xb') as f:
        f.write(header)
        points.tofile(f)
    with (args.out/name).open('rb') as f:
        assert f.read(len(header))==header and np.array_equal(np.fromfile(f,dtype),points)

save('raw.ply',cloud)
save('demo_edge3pct_conf20.ply',cloud[keep.ravel()])
save('removed_orange.ply',marked)
np.savez_compressed(args.out/'filter_masks.npz',keep=keep,edge=edge)
sheet=Image.new('RGB',(4*w,4*(h+22)),'white')
draw=ImageDraw.Draw(sheet)
rgb=np.stack([marked[k] for k in ('red','green','blue')],-1).reshape(n,h,w,3)
for i in range(n):
    u,v=i%4*w,i//4*(h+22)
    sheet.paste(Image.fromarray(rgb[i]),(u,v+22))
    draw.text((u+4,v+3),f'{i:02}  removed {(~keep[i]).mean():.1%}',fill='black')
sheet.save(args.out/'removed_pixels.jpg',quality=94)
result=dict(status='COMPLETED',source=str(args.p13),frames=n,raster=[h,w],raw_points=int(raw.size),
            kept=int(keep.sum()),removed=int((~keep).sum()),edge_removed=int(edge.sum()),
            confidence_only_removed=int((~keep&~edge).sum()),confidence_threshold=threshold,
            depth_filter='3x3 centered max-min relative jump >3%; confidence p20 after edge mask',
            original_geometry_recovery='P13 stored original depths and RGB PLY observation ids; infer each camera-origin from affine XYZ raster, undo P13 depths along rays with no new inference',
            maximum_affine_validation_error=max(fit_error))
(args.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
