"""Clip cache validation, scene splits, and training-only statistics."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_manifest(path):
    path = Path(path)
    document = json.loads(path.read_text())
    scenes, files, seen_frames = {}, set(), set()
    temporal = document.get('split_policy') == 'within_scene_temporal_pilot'
    for clip in document['clips']:
        scene,split = clip['scene'],clip['split']
        if split not in {'train','val','test'} or (not temporal and scenes.setdefault(scene,split)!=split):
            raise ValueError(f'Invalid split or scene leakage: {scene}')
        if temporal:
            frames = clip['source_frame_ids']
            identities = {(scene, frame) for frame in frames}
            if not identities or len(identities)!=len(frames) or identities & seen_frames:
                raise ValueError(f'Overlapping temporal pilot frames: {scene}')
            seen_frames.update(identities)
        cache = (path.parent/clip['cache']).resolve()
        if cache in files:
            raise ValueError(f'Duplicate clip cache: {cache}')
        files.add(cache)
        clip['cache']=str(cache)
    return document['clips']


def calibrated(gt, raw):
    y,x=np.indices(raw.shape[-2:])
    calibration=(x//16+3*(y//16))%5==1
    valid=np.isfinite(gt)&(gt>0)
    mask=valid&calibration
    if mask.sum()<12:
        raise ValueError('Need at least 12 positive sensor pixels in calibration blocks')
    return float(np.median(gt[mask]/raw[mask])), valid&~calibration


def stats(clips):
    values=[]
    for clip in clips:
        if clip['split']=='train':
            with np.load(clip['cache']) as cache:
                values.append(cache['confidence'].ravel()[::16])
    if not values:
        raise ValueError('No training clips')
    low,protect,high=np.quantile(np.concatenate(values),[.01,.70,.99])
    return dict(low=float(low),protect=float(protect),high=float(high))


class Clips(Dataset):
    def __init__(self, clips, split, with_gt, camera, neighbors='neighbors'):
        self.clips, self.frames = [], []
        for clip in clips:
            if clip['split']!=split:
                continue
            with np.load(clip['cache'], allow_pickle=False) as z:
                keys=['rgb','raw','confidence','sam','model_K','model_pose']
                a={k:z[k] for k in keys}
                a['static']=z['static'].astype(bool) if 'static' in z else np.ones_like(a['raw'],dtype=bool)
                for name in ('neighbors','eval_neighbors'):
                    a[name]=z[name] if name in z else np.full((len(a['raw']),2),-1,dtype=np.int32)
                if neighbors=='eval_neighbors' and np.any(
                    (a['neighbors'][:,:,None]==a['eval_neighbors'][:,None,:]) &
                    (a['eval_neighbors'][:,None,:]>=0)):
                    raise ValueError('Training and evaluation pairs must be disjoint')
                if 'wall_floor' in z and with_gt:
                    a['wall_floor']=z['wall_floor']
                if 'rule_depth' in z and with_gt:
                    a['rule_depth']=z['rule_depth']
                if camera=='provided':
                    for k in ['provided_K','provided_pose']:
                        a[k]=z[k].astype('float32')
                    if not with_gt:
                        if 'provided_scale' not in z:
                            raise ValueError('Provided pose inference needs an externally supplied provided_scale')
                        a['scale']=float(z['provided_scale'])
                if with_gt:
                    a['gt']=z['gt'].astype('float32')
                    if a['gt'].shape!=a['raw'].shape:
                        raise ValueError('GT raster must match raw')
                    a['scale'],a['test']=calibrated(a['gt'],a['raw'])
            size=a['raw'].shape
            if 'source_frame_ids' in clip and len(clip['source_frame_ids'])!=size[0]:
                raise ValueError('Source frame count does not match the cache')
            if (len(size)!=3 or min(size[1:])<8 or
                any(a[k].shape!=size for k in ['confidence','sam','static']) or
                a['rgb'].shape!=(*size,3) or a['rgb'].dtype!=np.uint8 or
                not np.isfinite(a['raw']).all() or np.min(a['raw'])<=0 or
                not np.isfinite(a['confidence']).all()):
                raise ValueError(f'Invalid clip raster: {clip["cache"]}')
            for key in ['wall_floor','rule_depth']:
                if key in a and a[key].shape!=size:
                    raise ValueError(f'Invalid {key} raster')
            for kind in ['model']+(['provided'] if camera=='provided' else []):
                if (a[f'{kind}_K'].shape!=(size[0],3,3) or a[f'{kind}_pose'].shape!=(size[0],4,4)
                    or not all(np.isfinite(a[f'{kind}_{key}']).all() for key in ('K','pose'))):
                    raise ValueError(f'Invalid {kind} camera arrays')
            for name in ('neighbors','eval_neighbors'):
                pairs=a[name]
                if pairs.shape!=(size[0],2) or not np.issubdtype(pairs.dtype,np.integer):
                    raise ValueError(f'{name} must be N×2 integer indices')
                if not ((pairs>=-1)&(pairs<size[0])).all():
                    raise ValueError(f'{name} index out of range')
                if np.any(pairs==np.arange(size[0])[:,None]):
                    raise ValueError('Neighbor must be another frame')
            a['clip']=clip
            a['normalizer']=float(np.median(a['raw']))
            a['raw']=a['raw'].astype('float32')
            idx=len(self.clips)
            self.frames.extend((idx,i) for i in range(size[0]))
            self.clips.append(a)
        self.camera=camera
        self.eval_pairs=neighbors=='eval_neighbors'

    def __len__(self):
        return len(self.frames)

    def __getitem__(self,index):
        idx,i=self.frames[index]
        a=self.clips[idx]
        def raster(key, at):
            return torch.from_numpy(np.array(a[key][at],copy=True)).unsqueeze(-3)
        def rgb(at):
            return torch.from_numpy(np.array(a['rgb'][at],copy=True)).movedim(-1,-3).float()/255
        sample=dict(raw=raster('raw',i),confidence=raster('confidence',i).float(),
                    sam=raster('sam',i).float(),static=raster('static',i),rgb=rgb(i),
                    normalizer=np.float32(a['normalizer']),clip=idx,frame=i)
        pair_sets=[('',a['neighbors'][i])]
        if self.eval_pairs:
            pair_sets.append(('eval_',a['eval_neighbors'][i]))
        for prefix, ids in pair_sets:
            ok=ids>=0
            ids=np.maximum(ids,0)
            sample.update({prefix+'source_rgb':rgb(ids),prefix+'source_raw':raster('raw',ids),
                           prefix+'source_static':raster('static',ids),
                           prefix+'has_source':torch.from_numpy(ok)})
            for kind in ['model']+(['provided'] if self.camera=='provided' else []):
                for key in ('K','pose'):
                    array=a[f'{kind}_{key}'].astype('float32')
                    if kind=='provided' and key=='pose':
                        array[:,:3,3]/=a['scale']
                    sample[f'{kind}_{key}']=torch.from_numpy(array[i])
                    sample[f'{prefix}{kind}_source_{key}']=torch.from_numpy(array[ids])
        if 'gt' in a:
            sample.update(gt=raster('gt',i),test=raster('test',i),scale=np.float32(a['scale']))
            sample['wall_floor']=raster('wall_floor',i) if 'wall_floor' in a else torch.zeros_like(sample['gt'],dtype=torch.uint8)
            sample['has_rule']='rule_depth' in a
            sample['rule_depth']=raster('rule_depth',i) if 'rule_depth' in a else sample['raw']
        return sample
