"""Create three independent synthetic plane clips for a pipeline smoke test."""
import json
from pathlib import Path
import sys

import numpy as np

out=Path(sys.argv[1])
out.mkdir(parents=True)
clips=[]
for scene,split in enumerate(['train','val','test']):
    n,h,w=4,48,64
    y,x=np.indices((h,w))
    K=np.array([[58,0,w/2],[0,58,h/2],[0,0,1]],dtype=np.float32)
    world_depth=2.+.2*scene
    pose=np.tile(np.eye(4,dtype=np.float32),(n,1,1))
    pose[:,0,3]=-np.arange(n)*.06
    coordinate=x[None]+(-pose[:,0,3,None,None])*K[0,0]/world_depth
    colors=np.broadcast_arrays(np.sin(coordinate/4)+1,np.cos(y[None]/5+coordinate/9)+1,np.sin(coordinate/7+scene)+1)
    rgb=(np.stack(colors,-1)*127).astype(np.uint8)
    gt=np.full((n,h,w),world_depth,dtype=np.float32)
    raw=gt*(1+.025*np.cos(coordinate/9)).astype('float32')
    conf=np.broadcast_to((coordinate%17).astype('float32'),raw.shape).copy()
    dist=np.broadcast_to(np.minimum(abs(x-w//2),8)/8,raw.shape).astype('float32')
    pairs={}
    for name,delta in [('neighbors',[1,-1]),('eval_neighbors',[2,-2])]:
        ids=np.arange(n)[:,None]+delta
        pairs[name]=np.where((ids>=0)&(ids<n),ids,-1).astype(np.int32)
    name=f'plane_{scene}.npz'
    np.savez_compressed(out/name,rgb=rgb,raw=raw,gt=gt,confidence=conf,sam=dist,
        model_K=np.tile(K,(n,1,1)),model_pose=pose,provided_K=np.tile(K,(n,1,1)),provided_pose=pose,**pairs)
    clips.append(dict(scene=f'synthetic-{scene}',split=split,cache=name))
(out/'dataset.json').write_text(json.dumps(dict(clips=clips),indent=2)+'\n')
