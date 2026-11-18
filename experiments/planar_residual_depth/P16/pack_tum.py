"""Pack TUM RGB-D, a frozen VGGT prediction, and SAM masks into a P16 cache."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def boundary(masks,h,w):
    edges=np.zeros((h,w),dtype=bool)
    if len(masks):
        if masks.shape[1:]!=(h,w):
            raise ValueError('SAM masks must match the model raster')
        edges[:,1:] |= (masks[:,:,1:]!=masks[:,:,:-1]).any(0)
        edges[:,:-1] |= (masks[:,:,1:]!=masks[:,:,:-1]).any(0)
        edges[1:] |= (masks[:,1:]!=masks[:,:-1]).any(0)
        edges[:-1] |= (masks[:,1:]!=masks[:,:-1]).any(0)
    if not edges.any():
        return np.ones((h,w),dtype=np.float32)
    return np.minimum(cv2.distanceTransform((~edges).astype('uint8'),cv2.DIST_L2,5),8)/8


def rows(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--masks',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--timestamps',type=Path,help='JSON array of exact TUM RGB timestamps in inference order')
    p.add_argument('--verify-input-manifest',type=Path,
                   help='VGGT inference manifest with the exact processed RGB hashes')
    p.add_argument('--interval',type=float,default=2)
    p.add_argument('--offset',type=float,default=0)
    p.add_argument('--rule',type=Path,help='Optional same-input P13 depths.npz')
    args=p.parse_args()
    z=np.load(args.raw,allow_pickle=False)
    raw=z['depth'][0,...,0]
    n,h,w=raw.shape
    rgb,depth=rows(args.dataset/'rgb.txt'),rows(args.dataset/'depth.txt')
    lookup=dict(rgb)
    stamps=json.loads(args.timestamps.read_text()) if args.timestamps else [
        min(rgb,key=lambda row:abs(float(row[0])-(float(rgb[0][0])+args.offset+i*args.interval)))[0]
        for i in range(n)]
    if len(stamps)!=n or len(set(stamps))!=n:
        raise ValueError('Need one distinct RGB timestamp per prediction')
    colors=[]; sam=[]; gt=[]; metadata=[]
    for i,stamp in enumerate(stamps):
        source=args.dataset/lookup[str(stamp)]
        image=Image.open(source).convert('RGB')
        if image.width*h!=image.height*w:
            raise ValueError('Packer supports same-aspect resize; cropped inputs need a custom cache')
        colors.append(np.array(image.resize((w,h),Image.Resampling.BICUBIC)))
        mask_path=args.masks/f'{i:02}.npz'
        masks=np.load(mask_path)['masks'].astype(bool)
        sam.append(boundary(masks,h,w))
        t,path=min(depth,key=lambda row:abs(float(row[0])-float(stamp)))
        gt_path=args.dataset/path
        if abs(float(t)-float(stamp))>.03:
            gt.append(np.zeros((h,w),dtype=np.float32))
            gt_path=None
        else:
            gt.append(np.array(Image.open(gt_path).resize((w,h),Image.Resampling.NEAREST),dtype=np.float32)/5000)
        metadata.append(dict(timestamp=stamp,rgb=str(source),rgb_sha256=digest(source),
                             sam=str(mask_path),sam_sha256=digest(mask_path),gt=str(gt_path) if gt_path else None))
    if args.verify_input_manifest:
        from io import BytesIO
        m=json.loads(args.verify_input_manifest.read_text())
        record=next(r for r in m['records'] if r['N']==n)
        observed=[]
        for color in colors:
            buffer=BytesIO()
            Image.fromarray(color).save(buffer,format='PNG')
            observed.append(hashlib.sha256(buffer.getvalue()).hexdigest())
        if observed!=record['input_sha256']:
            raise ValueError('Reconstructed RGB does not match frozen cache')
    trajectory=np.array(rows(args.dataset/'groundtruth.txt'),dtype=float)
    t=trajectory[:,0]
    target=np.array(stamps,dtype=float)
    model_pose=np.broadcast_to(np.eye(4,dtype=np.float32),(n,4,4)).copy()
    model_pose[:,:3]=z['extrinsics'][0]
    sets={}
    for name,offsets in [('neighbors',[1,-1]),('eval_neighbors',[2,-2])]:
        ids=np.arange(n)[:,None]+offsets
        sets[name]=np.where((ids>=0)&(ids<n),ids,-1).astype(np.int32)
    arrays=dict(rgb=colors,raw=raw,confidence=z['depth_conf'][0].reshape(raw.shape),
                sam=sam,gt=gt,model_pose=model_pose,model_K=z['intrinsics'][0],**sets)
    if target.min()>=t[0] and target.max()<=t[-1] and 'freiburg3' in args.dataset.name:
        orientation=Slerp(t,Rotation.from_quat(trajectory[:,4:8]))(target).as_matrix()
        center=np.stack([np.interp(target,t,trajectory[:,j]) for j in (1,2,3)],1)
        pose=np.broadcast_to(np.eye(4,dtype=np.float32),(n,4,4)).copy()
        pose[:,:3,:3]=orientation.transpose(0,2,1)
        pose[:,:3,3]=-(pose[:,:3,:3]@center[...,None])[...,0]
        k=np.array([[535.4*w/640,0,(320.1+.5)*w/640-.5],
                    [0,539.2*h/480,(247.6+.5)*h/480-.5],[0,0,1]],dtype='float32')
        arrays.update(provided_pose=pose,provided_K=np.broadcast_to(k,(n,3,3)).copy())
    if args.rule:
        rule=np.load(args.rule)
        if not np.array_equal(rule['raw'],raw):
            raise ValueError('Rule baseline depth does not match model cache')
        arrays['rule_depth']=rule['corrected']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('xb') as stream:
        np.savez_compressed(stream,**arrays)
    args.out.with_suffix('.json').write_text(json.dumps(dict(raw=str(args.raw),raw_sha256=digest(args.raw),
        rgb_alignment='verified' if args.verify_input_manifest else 'caller matched timestamps and preprocessing',
        frames=metadata),indent=2)+'\n')


if __name__=='__main__':
    main()
