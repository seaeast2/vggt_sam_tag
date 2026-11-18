"""Offline selective plane correction, with GT evaluation executed after freezing outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'results/planar_residual_depth'
SAM = BASE / 'P12-sam-indoor-20261118-001'
INPUT = BASE / 'P10-indoor-2s-input-20261118-001'
FULL = BASE / 'P10raw-three-sequences-20261118-003/indoor/predictions.npz'
EARLY = BASE / 'P12-context-20261118-001/indoor/N08/predictions.npz'
OUT = BASE / 'P12-indoor-20261118-001'
VIEWS = json.loads((INPUT / 'indoor/case.json').read_text())['images']

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def fit_plane(points):
    center = np.median(points, axis=0)
    keep = np.ones(len(points), dtype=bool)
    for _ in range(4):
        center = points[keep].mean(0)
        _, spread, vectors = np.linalg.svd(points[keep]-center, full_matrices=False)
        normal = vectors[-1]
        error = abs((points-center)@normal)
        keep = error <= max(np.quantile(error,.72),1e-6)
    return normal, float(normal@center), float(spread[1]/np.sqrt(keep.sum()))

def correct(path, name):
    pred = np.load(path)
    depths = pred['depth'][0,...,0]
    confidence = pred['depth_conf'][0].reshape(depths.shape)
    low, high = np.quantile(confidence,[.30,.70])
    output = depths.copy()
    full = depths.copy()
    selected = np.zeros_like(depths,dtype=bool)
    heldout = selected.copy()
    mask_ids = np.full(depths.shape,-1,dtype=np.int16)
    rows = []
    for i,(depth,conf,K) in enumerate(zip(depths,confidence,pred['intrinsics'][0])):
        y,x = np.indices(depth.shape)
        rays = np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(K).T
        points = rays*depth[...,None]
        test = ((x//8+3*(y//8))%5)==0
        masks = np.load(SAM/f'{i:02}.npz')['masks']
        for mid in sorted(range(len(masks)),key=lambda k: -int(masks[k].sum())):
            region = masks[mid]
            if region.sum()<600 or (region & (mask_ids[i]>=0)).sum()>.10*region.sum():
                continue
            if (region & (conf<=low)).sum()<120:
                continue
            support = region & ~test
            strong = support & (conf>=high)
            use_strong = int(strong.sum())>=120
            fit = strong if use_strong else support
            normal, offset, spread = fit_plane(points[fit])
            z = float(np.median(depth[fit]))
            # Camera-Y is only a coarse wall-orientation proxy.
            errors = abs(points[fit]@normal-offset)
            if abs(normal[1])>.55 or spread<.035*z or np.quantile(errors,.65)>.025*z:
                continue
            support_tiles = np.unique(np.c_[x[fit]//16,y[fit]//16],axis=0)
            if len(support_tiles)<6:
                continue
            den = rays@normal
            plane_depth = np.divide(offset,den,out=np.zeros_like(den),where=abs(den)>1e-6)
            valid = region & (plane_depth>0) & (mask_ids[i]<0)
            target = valid & (conf<=low)
            if (test & target).sum()<30 or valid.sum()<.97*region.sum():
                continue
            delta = np.clip(.5*(plane_depth-depth),-.05*depth,.05*depth)
            output[i][target] = (depth+delta)[target]
            full[i][valid] = (depth+delta)[valid]
            selected[i] |= target
            heldout[i] |= test & target
            mask_ids[i][valid] = mid
            h = test & target
            rows.append(dict(frame=i,mask=mid,mode='high_support' if use_strong else 'provisional',
                plane_normal=normal.tolist(),offset=offset,support_pixels=int(fit.sum()),
                support_tiles=len(support_tiles),modified=int(target.sum()),heldout=int(h.sum()),
                plane_distance_median_raw=float(np.median(abs(points[h]@normal-offset))),
                plane_distance_median_corrected=float(np.median(abs((rays[h]*output[i][h,None])@normal-offset)))))
    assert np.array_equal(output[~selected],depths[~selected])
    assert np.allclose(depths+(output-depths),output)
    np.savez_compressed(OUT/f'{name}.npz',raw=depths,corrected=output,residual=output-depths,
        whole_mask=full,selected=selected,heldout=heldout,mask_ids=mask_ids,confidence=confidence)
    dump(OUT/f'{name}.json',dict(low=float(low),high=float(high),regions=rows,
        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows

def evaluate():
    data=np.load(OUT/'full.npz')
    raw=data['raw']
    source=ROOT/'data/planar_residual_depth/P10/source/rgbd_dataset_freiburg3_structure_texture_far'
    index=[line.split() for line in (source/'depth.txt').read_text().splitlines() if line and not line.startswith('#')]
    gt=[]
    for v in VIEWS:
        t=float(v['timestamp'])
        stamp,file=min(index,key=lambda pair: abs(float(pair[0])-t))
        assert abs(float(stamp)-t)<.03
        gt.append(np.asarray(Image.open(source/file).resize((raw.shape[2],raw.shape[1]),Image.Resampling.NEAREST))/5000.)
    gt=np.stack(gt)
    y,x=np.indices(raw.shape[1:])
    calibration=((x//16+3*(y//16))%5)==1
    valid=(gt>0)&np.isfinite(raw)&(raw>0)
    cal=valid & calibration
    scale=float(np.median(gt[cal]/raw[cal]))
    test=data['heldout'] & valid & ~calibration
    rows=json.loads((OUT/'full.json').read_text())['regions']
    provisional=np.zeros_like(test)
    for row in rows:
        if row['mode']=='provisional':
            i=row['frame']
            provisional[i] |= data['mask_ids'][i]==row['mask']
    result={'evaluation_only_scale':scale,'metric':'pixel-weighted camera-Z MAE','groups':{}}
    for label,mask in [('all',test),('provisional',test&provisional),('high_support',test&~provisional)]:
        result['groups'][label]={'pixels':int(mask.sum()),'MAE_m':{
            k:float(np.mean(abs(scale*data[k][mask]-gt[mask]))) if mask.any() else None
            for k in ['raw','whole_mask','corrected']}}
    block_test=((x//8+3*(y//8))%5)==0
    mask_all=(data['mask_ids']>=0) & block_test & valid & ~calibration
    result['groups']['all_candidate_mask_pixels']={
        'pixels':int(mask_all.sum()),
        'MAE_m': {k:float(np.mean(abs(scale*data[k][mask_all]-gt[mask_all])))
                  if mask_all.any() else None for k in ['raw','whole_mask','corrected']}}
    dump(OUT/'evaluation.json',result)
    print(json.dumps(result,indent=2))

args=argparse.ArgumentParser()
args.add_argument('stage',choices=['fit','evaluate'])
if args.parse_args().stage=='fit':
    OUT.mkdir()
    dump(OUT/'protocol.json',dict(low_quantile=.30,high_quantile=.70,blend=.5,relative_cap=.05,
        fit_excludes='spatial 8px holdout blocks',gt_before_fit=False, scope='single-view plane support; provisional planes; first-camera Y orientation proxy'))
    for path,name in [(EARLY,'early'),(FULL,'full')]:
        rows=correct(path,name)
        print(name,len(rows),sum(r['mode']=='provisional' for r in rows),flush=True)
else:
    evaluate()
