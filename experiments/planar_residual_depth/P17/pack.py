"""Pack frozen model outputs and references into scene-separated P16 caches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser()
p.add_argument('prepared',type=Path)
p.add_argument('run',type=Path)
args=p.parse_args()
previous=ROOT/'results/planar_residual_depth/P16-real-20261118-001'
old=json.loads((previous/'dataset.json').read_text())['clips']
clips=[]
for row in old:
    if (row['domain'],row['split']) in {('outdoor','train'),('drone','train'),('indoor','test')}:
        clips.append(dict(row,cache=str(previous/row['cache'])))
for sample in json.loads((args.prepared/'samples.json').read_text()):
    name,split=sample['sample'],sample['split']
    source=args.prepared/name
    target=args.run/name
    with np.load(target/'vggt.npz') as z:
        raw,confidence,ext,k=[z[key] for key in ('raw','confidence','ext','K')]
    sam=np.load(target/'sam.npz')['sam']
    rgb=np.stack([np.asarray(Image.open(ROOT/v['processed']).convert('RGB')) for v in sample['images']])
    gt=np.load(source/'gt.npy')
    assert raw.shape==sam.shape==gt.shape==confidence.shape==rgb.shape[:3]
    n=len(raw)
    pose=np.tile(np.eye(4,dtype='float32'),(n,1,1))
    pose[:,:3]=ext
    pairs={}
    for key,offset in [('neighbors',1),('eval_neighbors',2 if n>2 else 1)]:
        ids=np.arange(n)[:,None]+np.array([offset,-offset])
        ids=np.where((ids>=0)&(ids<n),ids,-1).astype('int32')
        if key=='neighbors' and split!='train':
            ids[:]=-1
        pairs[key]=ids
    cache=target/'clip.npz'
    with cache.open('xb') as stream:
        np.savez_compressed(stream,rgb=rgb,raw=raw,confidence=confidence,sam=sam,gt=gt,
                            model_K=k,model_pose=pose,**pairs)
    clips.append(dict(scene=sample['scene'],split=split,domain=sample['domain'],cache=str(cache.relative_to(args.run)),
                      source_frame_ids=[v['source'] for v in sample['images']],
                      provenance=sample['provenance'],gt_sha256=hashlib.sha256((source/'gt.npy').read_bytes()).hexdigest()))
for name,selection in [('dataset',clips),('test-stereo',[c for c in clips if c['split']=='test' and c['domain']=='indoor-stereo']),
                       ('test-tum',[c for c in clips if c['split']=='test' and c['domain']=='indoor'])]:
    (args.run/f'{name}.json').write_text(json.dumps(dict(clips=selection),indent=2)+'\n')
print({s:sum(len(c['source_frame_ids']) for c in clips if c['split']==s) for s in ('train','val','test')},flush=True)
