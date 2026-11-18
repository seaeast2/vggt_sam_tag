"""Check P27 projection equations and independently orient a local RGB-D wall reference."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
p.add_argument('dataset',type=Path)
p.add_argument('out',type=Path)
a=p.parse_args()
a.out.mkdir(parents=True)
r=json.loads((a.run/'summary.json').read_text())
with np.load(a.run/'input.npz') as f:
    xyz,rgb,proj,raw=[f[k] for k in ('xyz','rgb','cameras','depth')]
with np.load(a.run/'wall_support.npz') as z:
    masks,selected=z['mask'],z['applied']
height,width=raw.shape[1:]
yy,xx=np.indices((height,width))
K=np.array([[535.4*width/640,0,(320.1+.5)*width/640-.5],
            [0,539.2*height/480,(247.6+.5)*height/480-.5],[0,0,1.]])
ray=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(K).T
def rows(file):
    return [s.split() for s in (a.dataset/file).read_text().splitlines() if s and not s.startswith('#')]
timestamps=rows('rgb.txt')
drows=rows('depth.txt')
start=float(timestamps[0][0])
def angle(x,y):
    return float(np.degrees(np.arccos(np.clip(abs(x@y),0,1))))
def fit(pts):
    use=pts
    for _ in range(5):
        c=use.mean(0)
        _,singular,v=np.linalg.svd(use-c,full_matrices=False)
        n=v[-1]
        d=float(c@n)
        err=abs(pts@n-d)
        use=pts[err<=max(float(np.quantile(err,.8)),.001)]
    return n,d,float(singular[1]/np.sqrt(len(use))),float(np.median(err))
result={}
for i in r['frames']:
    t,path=min(timestamps,key=lambda row:abs(float(row[0])-start-2*i))
    image=np.asarray(Image.open(a.dataset/path).convert('RGB').resize((width,height),Image.Resampling.BICUBIC))
    assert np.array_equal(image,rgb[i]),i
    stamp,gpath=min(drows,key=lambda row:abs(float(row[0])-float(t)))
    assert abs(float(stamp)-float(t))<=.03
    gt=np.asarray(Image.open(a.dataset/gpath).resize((width,height),Image.Resampling.NEAREST)).astype(float)/5000
    valid=masks[i]&(gt>0)
    pts=ray[valid]*gt[valid,None]
    item=dict(mask_pixels=int(masks[i].sum()),gt_pixels=len(pts),applied_pixels=int(selected[i].sum()),timestamp=t)
    if len(pts)>=100:
        normal,d,span,mad=fit(pts)
        item.update(reference_normal_camera=normal.tolist(),reference_offset_m=d,
                    reference_median_residual_m=mad,reference_second_axis_spread_m=span)
        _,R,*_=cv2.decomposeProjectionMatrix(proj[i])
        vals={}
        src=selected[i]
        world=xyz[i][src]
        pix=np.c_[xx[src],yy[src]]
        camera=proj[i]
        C=-np.linalg.solve(camera[:,:3],camera[:,3])
        rays=np.c_[pix,np.ones(len(pix))]@np.linalg.inv(camera[:,:3]).T
        for name,arm in r['arms'].items():
            m=arm['planes'][str(i)]
            n=np.array(m['normal'])
            offset=m['offset']
            z=(offset-C@n)/(rays@n)
            q=C+z[:,None]*rays
            projected=q@camera[:,:3].T+camera[:,3]
            shift=q-world
            norm=np.linalg.norm(shift,axis=1)
            vals[name]=dict(angle_to_sensor_wall_deg=angle(R@n,normal),
                max_pixel_reprojection_px=float(np.max(abs(projected[:,:2]/projected[:,2,None]-pix))),
                max_plane_residual=float(np.max(abs(q@n-offset))),
                median_angle_of_displacement_to_reference_normal_deg=float(np.degrees(np.arccos(
                  np.median(np.clip(abs((shift@R.T)@normal)/np.maximum(norm,1e-10),0,1))))))
        item['arms']=vals
    result[i]=item
(a.out/'diagnostic.json').write_text(json.dumps(dict(note='sensor depths used only for audit; identical original RGB verified',frames=result),indent=2)+'\n')
print(json.dumps(result,indent=2))
