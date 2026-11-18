"""Freeze RGB/GT sampling and physical-scene splits for P17."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'data/planar_residual_depth'
RESULTS=ROOT/'results/planar_residual_depth'
p=argparse.ArgumentParser()
p.add_argument('out',type=Path)
p.add_argument('--include-tum',nargs='*',default=[])
p.add_argument('--include-stereo',nargs='*',default=[])
args=p.parse_args()
args.out.mkdir(parents=True)
samples=[]

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def add(name,scene,split,frames,domain,source):
    folder=args.out/name
    folder.mkdir()
    views,gt=[],[]
    for i,(rgb,z,reference) in enumerate(frames):
        im=Image.open(rgb).convert('RGB')
        w,h=im.size
        cw=min(w,2*h)
        ch=min(h,2*w)
        x=(w-cw)//2
        y=(h-ch)//2
        if cw>=ch:
            size=(384,round(384*ch/cw/16)*16)
        else:
            size=(round(384*cw/ch/16)*16,384)
        image=im.crop((x,y,x+cw,y+ch)).resize(size,Image.Resampling.BICUBIC)
        path=folder/f'rgb_{i:02}.png'
        image.save(path)
        gt.append(np.array(Image.fromarray(z[y:y+ch,x:x+cw]).resize(size,Image.Resampling.NEAREST)))
        views.append(dict(processed=str(path.resolve().relative_to(ROOT)),size=list(size),
            source=str(rgb.relative_to(ROOT)),source_sha256=digest(rgb),reference=str(reference.relative_to(ROOT))))
    gt=np.asarray(gt,dtype=np.float32)
    if np.sum(np.isfinite(gt)&(gt>0))<1000:
        raise ValueError(f'Insufficient depth support for {name}')
    np.save(folder/'gt.npy',gt)
    samples.append(dict(sample=name,scene=scene,split=split,domain=domain,provenance=source,images=views))

def stereo(folder,scene,split):
    calibration=dict(x.split('=',1) for x in (folder/'calib.txt').read_text().splitlines() if '=' in x)
    fx=float(calibration['cam0'].lstrip('[').split()[0])
    frames=[]
    for i in (0,1):
        path=folder/f'disp{i}.pfm'
        if not path.exists():
            path=folder/f'disp{i}GT.pfm'
        disp=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if disp is None or disp.ndim!=2:
            raise ValueError(f'Could not read {path}')
        denominator=disp+float(calibration['doffs'])
        depth=np.divide(fx*float(calibration['baseline'])/1000,denominator,
                        out=np.zeros_like(disp),where=np.isfinite(disp)&(denominator>0))
        frames.append((folder/f'im{i}.png',depth,path))
    add('mb-'+folder.name, 'Middlebury-'+scene,split,frames,'indoor-stereo',
        'Middlebury calibrated disparity; camera-Z = fx*baseline/(disparity+doffs)')

mb=DATA/'middlebury/source/MiddEval3/trainingQ'
for folder in sorted(mb.iterdir()):
    scene=re.sub(r'^(Motorcycle|Piano|Playtable)[ELP]$',r'\1',folder.name)
    split='val' if scene in {'Piano','Recycle'} else 'test' if scene in {'Playroom','Teddy','Playtable'} else 'train'
    stereo(folder,scene,split)
for name in args.include_stereo:
    stereo(DATA/'P17/source'/f'{name}-perfect',name,'train')

def tuples(path):
    return [line.split() for line in path.read_text().splitlines() if line and not line.startswith('#')]
for name in args.include_tum:
    folder=DATA/'P17/source'/f'rgbd_dataset_{name}'
    rgb=tuples(folder/'rgb.txt')
    depth=tuples(folder/'depth.txt')
    times=np.array([float(t) for t,_ in rgb])
    depth_times=np.array([float(t) for t,_ in depth])
    split='val' if name=='freiburg2_desk' else 'train'
    # Freiburg1 recordings may share a physical office.
    scene='TUM-freiburg1-office' if name.startswith('freiburg1_') else f'TUM-{name}'
    starts=np.arange(times[0]+1,times[-1]-1.6,3)
    if len(starts)>16:
        starts=starts[np.linspace(0,len(starts)-1,16,dtype=int)]
    for j,start in enumerate(starts):
        frames=[]
        for t in start+.3*np.arange(6):
            i=int(np.argmin(abs(times-t)))
            d=int(np.argmin(abs(depth_times-times[i])))
            if abs(depth_times[d]-times[i])>.025:
                break
            gt=folder/depth[d][1]
            z=np.asarray(Image.open(gt),dtype=np.float32)/5000
            frames.append((folder/rgb[i][1],z,gt))
        if len(frames)==6:
            add(f'{name}-{j:02}',scene,split,frames,'indoor-video','TUM registered RGB-D; nearest timestamp ≤25ms')

(args.out/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
print('clips',len(samples),'frames',sum(len(c['images']) for c in samples))
