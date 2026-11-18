"""Apply paper methods to the frozen P20 TUM clip and export common-frame PLY."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/planar_residual_depth'
CACHE=BASE/'P16-real-20261118-001/indoor-test/clip.npz'

def flying(raw, rgb_native, k):
    """Port the public MATLAB SAD detector, FoV test and ray least squares."""
    h,w=raw.shape
    yy,xx=np.indices((h,w))
    rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(k).T
    xyz=rays*raw[...,None]
    sad=np.zeros_like(raw)
    for dy in range(-2,3):
        for dx in range(-2,3):
            y1,y2=max(0,-dy),min(h,h-dy)
            x1,x2=max(0,-dx),min(w,w-dx)
            sad[y1:y2,x1:x2]+=abs(raw[y1:y2,x1:x2]-raw[y1+dy:y2+dy,x1+dx:x2+dx])
    n=int(np.ceil(.05*raw.size))
    bad=np.zeros(raw.size,dtype=bool)
    bad[np.argsort(-sad.ravel(),kind='stable')[:n]]=True
    bad=bad.reshape(h,w)
    # Uniform stratified samples inside each RGB footprint on the native raster.
    nh,nw=rgb_native.shape[:2]
    patches=[]
    for ay in (1/6,.5,5/6):
        for ax in (1/6,.5,5/6):
            patches.append(cv2.remap(rgb_native.astype('float32'),((xx+ax)*nw/w-.5).astype('float32'),
                ((yy+ay)*nh/h-.5).astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE))
    patches=np.stack(patches,2)
    # Generalize fixed Oyla FoV to the available calibrated pixel grid.
    tangent=np.tan(5*np.arctan(1/np.diag(k)[:2]))
    result=raw.copy()
    for y,x in zip(*np.where(bad)):
        p=xyz[y,x]
        rho=np.linalg.norm(p)
        ex=np.array([rho*tangent[0],0,0])
        ey=np.array([0,rho*tangent[1],0])
        left,right,up,down=p-ex,p+ex,p+ey,p-ey
        normals=np.array([np.cross(left,down),np.cross(left,up),
                          np.cross(up,right),np.cross(right,down)])
        corners=np.stack([left+ey,left-ey,right+ey,right-ey])@k.T
        uv=corners[:,:2]/corners[:,2,None]
        x0,y0=np.maximum(0,np.floor(uv.min(0)).astype(int))
        x1,y1=np.minimum([w,h],np.ceil(uv.max(0)).astype(int)+1)
        support=xyz[y0:y1,x0:x1]
        good=(~bad[y0:y1,x0:x1])&np.all((support@normals.T)*(normals@p)>0,axis=-1)
        if not good.any():
            continue
        colors=patches[y0:y1,x0:x1][good]
        mismatch=np.linalg.norm(colors-patches[y,x],axis=-1).mean(1)**2
        weights=np.exp(-mismatch/(max(float(mismatch.max()),1e-12)*.02))
        # Orthogonal projection of the weighted 3D centroid onto the source ray.
        unit=p/rho
        z=float(np.average(support[good]@unit,weights=weights)*unit[2])
        if np.isfinite(z) and z>0:
            result[y,x]=z
    return result


def align_global(output,raw,confidence):
    valid=np.isfinite(output)&(output>0)&(confidence>=np.quantile(confidence,.7))
    factor=float(np.median(raw[valid]/output[valid]))
    return output*factor,factor


def run_model(args,cache):
    rgb,raw=cache['rgb'],cache['raw']
    if args.method=='raw':
        return raw,dict(interpretation='original VGGT-Omega')
    if args.method=='svm':
        path=BASE/'P20-boundary-svm-20261118-001/indoor-test/depths.npz'
        z=np.load(path)
        assert np.array_equal(z['raw'],raw)
        return z['corrected'],dict(interpretation='P20 SVM')
    if args.method=='flying':
        case=json.loads((CACHE.parent/'case.json').read_text())
        output=[flying(z,np.array(Image.open(ROOT/v['source']).convert('RGB')),k)
                for z,k,v in zip(raw,cache['model_K'],case['images'])]
        return np.stack(output),dict(interpretation='Python port of author SAD/FoV/weighted-ray method',
            adaptation='VGGT K, native RGB stratified inside each footprint; omit 0.01 physical-unit bias in model gauge')
    import torch
    if args.method=='layered':
        from segment_anything import SamAutomaticMaskGenerator,sam_model_registry
        generator=SamAutomaticMaskGenerator(sam_model_registry['vit_b'](
            checkpoint=str(ROOT/'models/sam/sam_vit_b_01ec64.pth')).cuda().eval(),
            points_per_side=16,pred_iou_thresh=.85,stability_score_thresh=.9)
        masks=[]
        for image,z in zip(rgb,raw):
            proposals=generator.generate(image)
            median=np.median(z)
            eligible=[m for m in proposals if .03*z.size<=m['area']<=.65*z.size
                      and np.median(z[m['segmentation']])<median]
            if not eligible:
                raise ValueError('No foreground SAM proposal met the frozen eligibility rule')
            masks.append(max(eligible,key=lambda m:m['area'])['segmentation'])
        del generator
        torch.cuda.empty_cache()
        sys.path.insert(0,str(args.repo))
        from MiT_siamese import RelDepthModel
        model=RelDepthModel().eval()
        model.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True),strict=True)
        model.cuda()
        outputs=[]
        for i,(image,z,mask) in enumerate(zip(rgb,raw,masks)):
            h,w=z.shape
            white=float(z.max())
            inverse=1-z/white
            shape=(320,320)
            small=cv2.resize(inverse,shape,interpolation=cv2.INTER_AREA)
            image=cv2.resize(image,shape,interpolation=cv2.INTER_AREA).astype('float32')/255
            image=((image-np.array([.485,.456,.406]))/np.array([.229,.224,.225])).astype('float32')
            def tensor(array):
                return torch.from_numpy(array.astype('float32'))[None].cuda()
            with torch.inference_mode():
                _,a,b=model({'input_rgb':tensor(image.transpose(2,0,1)),
                    'input_depth':tensor((small/max(float(small.max()),1e-8)*10)[None]),
                    'mask':tensor(cv2.resize(mask.astype('float32'),shape,interpolation=cv2.INTER_AREA)[None])})
            fg,bg=[cv2.resize(t.squeeze().float().cpu().numpy().clip(0,10)/10,(w,h)) for t in (a,b)]
            outputs.append(np.maximum((1-np.where(mask,fg,bg))*white,1e-6))
            Image.fromarray(mask.astype('uint8')*255).save(args.out/f'mask_{i:02}.png')
        return np.stack(outputs),dict(interpretation='official checkpoint and layered model; one SAM foreground mask per view',
            adaptation='official normalized depth encoding and high-resolution composite, decoded using input depth maximum; one largest eligible SAM proposal')
    if args.method=='depth-pro':
        sys.path.insert(0,str(args.repo/'src'))
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        cfg=replace(DEFAULT_MONODEPTH_CONFIG_DICT,checkpoint_uri=str(args.weights))
        model,transform=depth_pro.create_model_and_transforms(config=cfg,device=torch.device('cuda'),precision=torch.float16)
        model.eval()
        outputs=[]
        for image in rgb:
            with torch.inference_mode():
                ans=model.infer(transform(Image.fromarray(image)))
            outputs.append(ans['depth'].float().cpu().numpy())
        output,scale=align_global(np.stack(outputs),raw,cache['confidence'])
        return output,dict(interpretation='official RGB-only Depth Pro new predictions; not VGGT residual refinement',
                           model_gauge_scale=scale)
    raise ValueError(args.method)


def export(out,cache,z):
    n,h,w=z.shape
    y,x=np.indices((h,w))
    rays=np.stack([x,y,np.ones_like(x)],-1)
    dtype=np.dtype([(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')]+
                   [('confidence','<f4')]+[(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])
    records=np.empty(n*h*w,dtype=dtype)
    for i in range(n):
        pose=cache['model_pose'][i]
        inv=np.linalg.inv(cache['model_K'][i])
        xyz=(((rays@inv.T)*z[i,...,None]-pose[:3,3])@np.linalg.inv(pose[:3,:3]).T).reshape(-1,3)
        assert np.isfinite(xyz).all()
        b=records[i*h*w:(i+1)*h*w]
        for j,key in enumerate('xyz'):
            b[key]=xyz[:,j]
        for j,key in enumerate(('red','green','blue')):
            b[key]=cache['rgb'][i,...,j].ravel()
        b['frame_id'],b['pixel_u'],b['pixel_v']=i,x.ravel(),y.ravel()
        b['confidence']=cache['confidence'][i].ravel()
    header=(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(records)}\n'
            'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n'
            'property float confidence\nproperty ushort frame_id\nproperty ushort pixel_u\nproperty ushort pixel_v\nend_header\n').encode()
    with (out/'sample.ply').open('xb') as f:
        f.write(header)
        records.tofile(f)
    with (out/'sample.ply').open('rb') as f:
        assert f.read(len(header))==header and np.array_equal(np.fromfile(f,dtype),records)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('method',choices=['raw','svm','flying','layered','depth-pro'])
    p.add_argument('out',type=Path)
    p.add_argument('--repo',type=Path)
    p.add_argument('--weights',type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True)
    with np.load(CACHE) as f:
        cache={k:f[k] for k in ('rgb','raw','model_K','model_pose','confidence')}
    z,info=run_model(args,cache)
    assert z.shape==cache['raw'].shape and np.isfinite(z).all() and (z>0).all()
    export(args.out,cache,z)
    np.save(args.out/'depth.npy',z)
    info.update(status='COMPLETED',frames=len(z),vertices=z.size,source=str(CACHE.relative_to(ROOT)))
    if args.weights:
        with args.weights.open('rb') as f:
            info['checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    (args.out/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    print(args.method,info,flush=True)


if __name__=='__main__':
    main()
