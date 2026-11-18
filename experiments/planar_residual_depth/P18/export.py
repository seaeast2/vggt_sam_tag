"""Export matched, full-clip P17 point clouds in the original model gauge."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('p17', type=Path)
p.add_argument('out', type=Path)
args = p.parse_args()
args.out.mkdir(parents=True)
dtype = np.dtype([(k,'<f4') for k in 'xyz'] + [(k,'u1') for k in ('red','green','blue')]
                 + [('confidence','<f4')] + [(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])

def read_npz(path):
    with np.load(path, allow_pickle=False) as file:
        return dict(file)

def write(path, records):
    header = (f'ply\nformat binary_little_endian 1.0\nelement vertex {len(records)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'property float confidence\nproperty ushort frame_id\n'
              'property ushort pixel_u\nproperty ushort pixel_v\nend_header\n').encode()
    with path.open('xb') as stream:
        stream.write(header)
        records.tofile(stream)
    with path.open('rb') as stream:
        assert stream.read(len(header)) == header
        assert np.array_equal(np.fromfile(stream,dtype),records)
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

manifest = []
for test in ('stereo','tum'):
    sensor = args.p17/f'eval-sensor-{test}'
    multiview = args.p17/f'eval-multiview-{test}'
    for folder in sorted(sensor.glob('clip_*')):
        other = multiview/folder.name
        source = json.loads((folder/'source.json').read_text())
        assert source == json.loads((other/'source.json').read_text())
        cache_path = Path(source['cache'])
        cache = read_npz(cache_path)
        output = args.out/((cache_path.parent.name.removeprefix('mb-')) if test=='stereo' else 'TUM')
        output.mkdir()
        frames = sorted(folder.glob('*.npz'))
        assert len(frames) == len(cache['raw'])
        arms = {k:[] for k in ('raw','sensor','multiview')}
        for i,path in enumerate(frames):
            assert i==int(path.stem)
            a,b = read_npz(path),read_npz(other/path.name)
            for key in ('K','pose','raw'):
                assert np.array_equal(a[key],b[key])
            assert np.array_equal(a['raw'],cache['raw'][i])
            z = a['raw']
            y,x = np.indices(z.shape)
            rays = np.stack([x,y,np.ones_like(x)],-1) @ np.linalg.inv(a['K'].astype(float)).T
            pose = a['pose'].astype(float)
            inv_r = np.linalg.inv(pose[:3,:3])
            colors = cache['rgb'][i]
            obs = np.empty(z.size,dtype)
            for k,j in zip(('red','green','blue'),range(3)):
                obs[k] = colors[...,j].ravel()
            obs['confidence'] = cache['confidence'][i].ravel()
            obs['frame_id'],obs['pixel_u'],obs['pixel_v'] = i,x.ravel(),y.ravel()
            for name,depth in [('raw',z),('sensor',a['corrected']),('multiview',b['corrected'])]:
                assert np.isfinite(depth).all() and (depth>0).all()
                xyz = ((rays*depth[...,None]-pose[:3,3])@inv_r.T).reshape(-1,3)
                assert np.isfinite(xyz).all()
                points=obs.copy()
                for j,k in enumerate('xyz'):
                    points[k]=xyz[:,j]
                arms[name].append(points)
        files=[]
        for name,blocks in arms.items():
            records=np.concatenate(blocks)
            path=output/f'{name}.ply'
            files.append(dict(file=str(path.relative_to(args.out)),points=len(records),sha256=write(path,records)))
        manifest.append(dict(scene=source['scene'],source=source,frames=len(frames),files=files))
        print(output.name,len(frames),len(records),flush=True)
(args.out/'manifest.json').write_text(json.dumps(dict(source_run=str(args.p17.resolve()),
    coordinates='shared VGGT-Omega model frame, no metric scale applied',scenes=manifest),indent=2)+'\n')
