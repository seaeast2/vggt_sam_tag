"""Export KITTI depth-edge and confidence sweeps with identical point provenance."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'results/planar_residual_depth/P16-real-20261118-001/outdoor-test/clip.npz'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'P16'))
from data import calibrated
p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
args=p.parse_args()

DTYPE=np.dtype([(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')]
               +[('confidence','<f4')]+[(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])

def write(file,points):
    header=(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            'property float confidence\nproperty ushort frame_id\n'
            'property ushort pixel_u\nproperty ushort pixel_v\nend_header\n').encode()
    with file.open('xb') as stream:
        stream.write(header)
        points.tofile(stream)
    with file.open('rb') as stream:
        assert stream.read(len(header))==header
        assert np.array_equal(np.fromfile(stream,DTYPE),points)

def relative_jump(raw,kernel):
    # The official demo defines both extremum filters with edge padding.
    shape=np.ones((kernel,kernel),dtype=np.uint8)
    return np.array([(cv2.dilate(z,shape,borderType=cv2.BORDER_REPLICATE)-
                      cv2.erode(z,shape,borderType=cv2.BORDER_REPLICATE))/
                     np.maximum(abs(z),1e-6) for z in raw])

def cloud(cache,scale):
    raw,conf=cache['raw'],cache['confidence']
    n,h,w=raw.shape
    y,x=np.indices((h,w))
    uv=np.stack([x,y,np.ones_like(x)],-1)
    records=np.empty(n*h*w,dtype=DTYPE)
    reference=cache['model_pose'][0].astype(float)
    for j,(z,k,pose) in enumerate(zip(raw,cache['model_K'],cache['model_pose'])):
        r,t=pose[:3,:3].astype(float),pose[:3,3].astype(float)
        world=((uv@np.linalg.inv(k.astype(float)).T)*z[...,None]-t)@np.linalg.inv(r).T
        xyz=(world@reference[:3,:3].T+reference[:3,3])*scale
        assert np.isfinite(xyz).all()
        block=records[j*h*w:(j+1)*h*w]
        for i,axis in enumerate('xyz'):
            block[axis]=xyz[...,i].ravel()
        for i,channel in enumerate(('red','green','blue')):
            block[channel]=cache['rgb'][j,...,i].ravel()
        block['confidence']=conf[j].ravel()
        block['frame_id'],block['pixel_u'],block['pixel_v']=j,x.ravel(),y.ravel()
    return records

def to_base(a,size,interpolation=cv2.INTER_NEAREST):
    if a.dtype==bool:
        a=a.astype('uint8')
    return np.stack([cv2.resize(z,size,interpolation=interpolation) for z in a])

base=dict(np.load(SOURCE))
ref=base['raw']
sensor_scale,test=calibrated(base['gt'],ref)
n,h0,w0=ref.shape
summary={}
reference_test={}
for width in (384,512,640):
    folder=args.run/f'R{width}'
    folder.mkdir(exist_ok=True)
    if width==384:
        cache=base
    elif not (folder/'prediction.npz').is_file():
        continue
    else:
        with np.load(folder/'prediction.npz') as f:
            cache=dict(f)
    raw,conf=cache['raw'],cache['confidence']
    assert raw.shape==conf.shape and np.isfinite(raw).all() and np.isfinite(conf).all()
    assert (raw>0).all()
    mapped=to_base(raw,(w0,h0),cv2.INTER_LINEAR)
    # Align one global scale using the first-frame raw models, without sensor GT.
    ratio=1. if width==384 else float(np.median(ref[0]/mapped[0]))
    reference_test[width]=abs(mapped*ratio*sensor_scale-base['gt'])
    points=cloud(cache,ratio)
    jump=relative_jump(raw,3)
    edge=jump>.03
    conf20=np.quantile(conf,.20)
    masked_conf=np.where(edge,0,conf)
    demo_thresh=np.quantile(masked_conf,.20)
    options={
        'raw':np.ones(raw.shape,dtype=bool),
        'edge_3x3_1p5pct':jump<=.015,
        'edge_3x3_3pct':~edge,
        'edge_3x3_5pct':jump<=.05,
        'edge_5x5_3pct':relative_jump(raw,5)<=.03,
        'confidence_p20':(conf>=conf20)&(conf>1e-5),
        'demo_edge3pct_conf20':(masked_conf>=demo_thresh)&(masked_conf>1e-5)}
    summary[width]=dict(first_frame_scale_to_R384=ratio,raster=list(raw.shape),options={})
    for name,keep in options.items():
        write(folder/f'{name}.ply',points[keep.ravel()])
        gt_keep=test & to_base(keep,(w0,h0)).astype(bool)
        stats=dict(kept=int(keep.sum()),removed=int((~keep).sum()),retained_fraction=float(keep.mean()),
                   retained_GT_pixels=int(gt_keep.sum()),
                   MAE_m_on_retained_GT_only=float(reference_test[width][gt_keep].mean()) if gt_keep.any() else None)
        summary[width]['options'][name]=stats
    marked=points.copy()
    for channel,value in zip(('red','green','blue'),(255,145,20)):
        marked[channel][edge.ravel()]=value
    write(folder/'edge_3pct_removed_orange.ply',marked)
    del marked
    hh,ww=raw.shape[-2:]
    grid=Image.new('RGB',(ww*3,(hh+23)*2),'white')
    draw=ImageDraw.Draw(grid)
    for i,image in enumerate(cache['rgb']):
        arr=image.copy()
        arr[edge[i]]=[255,145,20]
        x,y=i%3*ww,i//3*(hh+23)
        grid.paste(Image.fromarray(arr),(x,y+23))
        draw.text((x+4,y+3),f'{i:02}  3% edge mask | removed {edge[i].mean():.1%}',fill='black')
    grid.save(folder/'edge_mask_review.jpg',quality=94)
    print(width,{k:round(v['retained_fraction']*100,2) for k,v in summary[width]['options'].items()},flush=True)

case=json.loads((SOURCE.parent/'case.json').read_text())
meta=dict(status='COMPLETED',frames=[dict(source=v['source'],timestamp=v['timestamp'],sha256=v['source_sha256'])
                                   for v in case['images']],resolution=summary,
    comparison=dict(common_GT_pixels=int(test.sum()),raw_MAE_m={k:float(e[test].mean()) for k,e in reference_test.items()},
                    scale_R384_from_GT_calibration=sensor_scale,
                    note='higher resolution depth bilinearly sampled on original 384 evaluation grid; one scale per resolution from the first raw frame before sensor calibration'),
    notes=['Filters only remove points. Orange diagnostic retains those points and marks them.',
           'All frames use first camera coordinates and one clip scale. Each resolution has its own predicted K and pose.',
           'Demo-style filter includes edge and confidence rules; no sky filter or point-budget downsampling.',
           'Retained-GT MAE varies with coverage; evaluate it together with retained pixel count.'])
(args.run/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
