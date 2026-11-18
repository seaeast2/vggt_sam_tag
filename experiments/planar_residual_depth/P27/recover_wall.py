"""Estimate wall orientation from RGB lines and export a diagnostic projection."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

p=argparse.ArgumentParser()
p.add_argument('parent',type=Path)
p.add_argument('out',type=Path)
p.add_argument('--raw-anchor',action='store_true')
args=p.parse_args()
out=args.out
out.mkdir(parents=True)
with np.load(args.parent/'input.npz') as f:
    rgb,xyz,conf,keep,P,raw=[f[k] for k in ('rgb','xyz','confidence','keep','cameras','depth')]
with np.load(args.parent/'wall_support.npz') as f:
    masks=f['mask']
_,h,w=raw.shape
yy,xx=np.indices((h,w))
bottom={12:((320,86),(379,92)),13:((299,74),(378,84)),
        14:((284,68),(380,81)),15:((264,73),(335,85))}
vertical={12:((361,12),(349,76)),13:((377,13),(363,66)),
          14:((334,12),(324,62)),15:((344,13),(327,70))}
ids=list(bottom)

def locate(image, endpoints):
    gray=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(float),(3,3),.6)
    start,end=np.array(endpoints,float)
    direction=end-start
    normal=np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
    base=start+np.linspace(0,1,45)[:,None]*direction
    offset=np.arange(-7,7.1,.5)
    q=base[:,None]+offset[None,:,None]*normal
    values=cv2.remap(gray,q[:,:,0].astype('float32'),q[:,:,1].astype('float32'),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    points=q[np.arange(len(base)),values.argmin(1)]
    use=np.ones(len(points),dtype=bool)
    for _ in range(5):
        c=points[use].mean(0)
        _,_,v=np.linalg.svd(points[use]-c,full_matrices=False)
        n=v[-1]
        error=abs((points-c)@n)
        use=error<max(1.,float(np.percentile(error,65)))
    return np.r_[n,-n@c],points[use]

samples={}
for i in ids:
    samples[i]=[]
    for seeds in (bottom,vertical):
        line,points=locate(rgb[i],seeds[i])
        plane=line@P[i]
        samples[i].append((plane/np.linalg.norm(plane[:3]),points))

def estimate(frames):
    B,V=[np.stack([samples[i][k][0] for i in frames]) for k in (0,1)]
    b,v=[np.linalg.svd(r[:,:3],full_matrices=False)[2][-1] for r in (B,V)]
    normal=np.cross(b,v)
    normal/=np.linalg.norm(normal)
    origin=np.linalg.lstsq(np.vstack([B[:,:3],b]),np.r_[-B[:,3],0.],rcond=None)[0]
    return normal,float(origin@normal),b,v

def angle(a,b):
    return float(np.degrees(np.arccos(np.clip(abs(a@b),0,1))))

n,d,b,v=estimate(ids)
leaveout={}
for omit in ids:
    normal,offset,*_=estimate([i for i in ids if i!=omit])
    leaveout[omit]=dict(normal_change_deg=angle(normal,n),
                        offset_change_model_units=float(abs(offset*np.sign(n@normal)-d)))
line_d=d
anchor_offsets={}
if args.raw_anchor:
    for i in ids:
        region=cv2.erode(masks[i].astype('uint8'),np.ones((3,3),dtype=np.uint8)).astype(bool)&keep[i]
        confidence=conf[i][region]
        reliable=region&(conf[i]>=np.median(confidence))
        anchor_offsets[i]=float(np.median(xyz[i][reliable]@n))
    d=float(np.median(list(anchor_offsets.values())))
eligible=np.zeros_like(keep)
projection=raw.copy()
points=xyz.copy()
for i in ids:
    panel=cv2.erode(masks[i].astype('uint8'),np.ones((3,3),dtype=np.uint8)).astype(bool)
    C=-np.linalg.solve(P[i,:3,:3],P[i,:,3])
    ray=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(P[i,:3,:3]).T
    denom=ray@n
    z=np.divide(d-C@n,denom,out=np.zeros_like(raw[i],dtype=float),where=abs(denom)>1e-8)
    good=panel&keep[i]&(z>0)&(abs(z-raw[i])<=.30*raw[i])
    eligible[i]=good
    projection[i][good]=z[good]
    points[i][good]=C+z[good,None]*ray[good]
    assert np.max(abs(points[i][good]@n-d))<1e-6
changed=(points!=xyz).any(axis=-1)
assert not (changed&~eligible).any()
assert np.isfinite(points).all()
dtype=np.dtype([(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')]+
               [('confidence','<f4')]+[(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])

def save(name,coords,visibility,orange):
    frame,y,x=np.where(visibility)
    array=np.empty(len(x),dtype)
    colors=rgb[visibility].copy()
    if orange:
        colors[changed[visibility]]=[255,145,20]
    for i,k in enumerate('xyz'):
        array[k]=coords[visibility][:,i]
    for i,k in enumerate(('red','green','blue')):
        array[k]=colors[:,i]
    array['confidence']=conf[visibility]
    array['frame_id'],array['pixel_u'],array['pixel_v']=frame,x,y
    head=(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(x)}\n'
          'property float x\nproperty float y\nproperty float z\nproperty uchar red\n'
          'property uchar green\nproperty uchar blue\nproperty float confidence\n'
          'property ushort frame_id\nproperty ushort pixel_u\nproperty ushort pixel_v\nend_header\n').encode()
    path=out/f'{name}.ply'
    with path.open('xb') as stream:
        stream.write(head)
        array.tofile(stream)
    with path.open('rb') as stream:
        assert stream.read(len(head))==head and np.array_equal(np.fromfile(stream,dtype),array)
save('line_wall_orange_full',points,keep,True)
save('line_wall_orange_wall_only',points,keep&masks,True)
save('raw_wall_only',xyz,keep&masks,False)
np.savez_compressed(out/'depths.npz',raw=raw,corrected=projection,changed=changed,keep=keep)
sheet=Image.new('RGB',(2*w,2*(h+24)),'white')
draw=ImageDraw.Draw(sheet)
for j,i in enumerate(ids):
    im=Image.fromarray(rgb[i])
    pen=ImageDraw.Draw(im)
    for k,color in enumerate(['cyan','orange']):
        for x,y in samples[i][k][1]:
            pen.point((int(round(x)),int(round(y))),fill=color)
    x,y=j%2*w,j//2*(h+24)
    sheet.paste(im,(x,y+24))
    draw.text((x+4,y+5),f'Frame {i} | cyan: lower seam, orange: vertical seam',fill='black')
sheet.save(out/'rgb_lines.jpg',quality=94)
result=dict(status='EXPLORATORY_RGB_LINE_PROJECTION',parent=str(args.parent),
    normal=n.tolist(),offset=d,rgb_line_offset=line_d,raw_anchor_offsets=anchor_offsets,lower_vertical_angle_deg=angle(b,v),leave_one_view_out=leaveout,
    guaranteed_stable=False,source='RGB manual line seeds localized to dark seams, fixed VGGT cameras; no sensor in fit',
    applied_pixels=int(changed.sum()),per_frame={i:int(changed[i].sum()) for i in ids},
    retained_points=int(keep.sum()),filter='frozen edge_3x3_3pct',depth_change_cap=.30,
    seeds=dict(bottom=bottom,vertical=vertical))
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
