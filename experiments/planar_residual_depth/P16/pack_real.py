"""Pack three aligned sensor references for the within-scene temporal pilot."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/planar_residual_depth/P10'))
from evaluate import make_GT
from pack_tum import digest

p = argparse.ArgumentParser()
p.add_argument('run',type=Path)
args=p.parse_args()
clips=[]
for domain in ('indoor','outdoor','drone'):
    for split in ('train','val','test'):
        folder=args.run/f'{domain}-{split}'
        case=json.loads((folder/'case.json').read_text())
        views=case['images']
        n=len(views)
        with np.load(folder/'vggt.npz') as z:
            raw,confidence,ext,k=[z[key] for key in ('raw','confidence','ext','K')]
        with np.load(folder/'sam.npz') as z:
            sam=z['sam']
        rgb=np.stack([np.asarray(Image.open(ROOT/v['processed']).convert('RGB')) for v in views])
        assert raw.shape==confidence.shape==sam.shape==rgb.shape[:3]
        assert np.isfinite(raw).all() and (raw>0).all()
        case['polygons']=[[(0,0),(w-1,0),(w-1,h-1),(0,h-1)] for w,h in (v['size'] for v in views)]
        gt,_=make_GT(case,folder)
        provenance=folder/'GT_provenance.json'
        reference=json.loads(provenance.read_text())
        reference.pop('depth_point_GT_used_for_fit',None)
        reference.pop('provided_pose_values_were_camera_control',None)
        reference.update(depth_role='sensor supervision for train; evaluation reference for val/test',
                         projection_uses_calibration=True,
                         provided_pose_used_for_scan_projection=domain=='drone')
        provenance.write_text(json.dumps(reference,indent=2)+'\n')
        pose=np.tile(np.eye(4,dtype='float32'),(n,1,1))
        pose[:,:3]=ext
        pairs={}
        for name,delta in [('neighbors',[1,-1]),('eval_neighbors',[2,-2])]:
            ids=np.arange(n)[:,None]+delta
            pairs[name]=np.where((ids>=0)&(ids<n),ids,-1).astype('int32')
        case_file=folder/'case.json'
        target=folder/'clip.npz'
        with target.open('xb') as stream:
            np.savez_compressed(stream,rgb=rgb,raw=raw,confidence=confidence,sam=sam,
                gt=gt,model_pose=pose,model_K=k,
                provided_pose=np.linalg.inv(np.array([v['T_WC'] for v in views])).astype('float32'),
                provided_K=np.array([v['K'] for v in views],dtype='float32'),**pairs)
        clips.append(dict(scene=case['metadata']['dataset'],domain=domain,split=split,
            cache=str(target.relative_to(args.run)),source_frame_ids=[v['timestamp'] for v in views],
            case_sha256=digest(case_file),valid_gt_pixels=int((gt>0).sum())))
        print(domain,split,clips[-1]['valid_gt_pixels'],flush=True)
manifest=dict(split_policy='within_scene_temporal_pilot',clips=clips,
              scope='Three training domains, one physical scene per domain. Validation and test use disjoint temporal blocks within each scene.')
(args.run/'dataset.json').write_text(json.dumps(manifest,indent=2)+'\n')
for domain in ('indoor','outdoor','drone'):
    selected=[c for c in clips if c['domain']==domain and c['split']=='test']
    (args.run/f'test-{domain}.json').write_text(json.dumps(dict(clips=selected),indent=2)+'\n')
