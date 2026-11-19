"""Render a registered RGB-D pointmap from the tracked frames of a P32 replay."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from replay import registered_depth


def draw(ax, points, colors, path, right, up, title, draw_path=False):
    right = np.asarray(right, dtype=float)
    right /= np.linalg.norm(right)
    up = np.asarray(up, dtype=float)
    up -= (up @ right) * right
    up /= np.linalg.norm(up)
    order = np.argsort(points @ np.cross(right, up))
    projected = points[order] @ np.column_stack((right, up))
    ax.set_facecolor('#111820')
    ax.scatter(*projected.T, c=colors[order], s=.85, linewidths=0)
    if draw_path:
        trajectory = path @ np.column_stack((right, up))
        ax.plot(*trajectory.T, color='#56e4f2', linewidth=1.5, zorder=4)
        ax.scatter(*trajectory.T, color='#56e4f2', s=14, zorder=5)
        ax.scatter(*trajectory[:1].T, color='#f9cb66', s=42, zorder=6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, color='#d4e0ec', loc='left', fontsize=13, pad=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('image', type=Path)
    parser.add_argument('--mode', choices=('full', 'patch', 'patch_fallback'), default='full')
    parser.add_argument('--ply-only', action='store_true',
                        help='Write every registered RGB-D sample as a binary PLY to the output path')
    args = parser.parse_args()
    replay = json.loads((args.run / 'replay.json').read_text())
    if 'rgb_input' in replay:
        rgb_input, manifest = replay['rgb_input'], replay['manifest']
    else:
        mapping = json.loads((Path(replay['source']) / 'map.json').read_text())
        rgb_input, manifest = mapping['source'], mapping['manifest']
    with np.load(rgb_input) as source:
        rgb = source['rgb']
    depth, k = registered_depth(Path(manifest), rgb.shape[1:3])
    poses = np.load(args.run / f'{args.mode}_poses.npy')
    y, x = np.indices(depth.shape[1:])
    rays = np.stack((x,y,np.ones_like(x)),axis=-1) @ np.linalg.inv(k).T
    cloud, colors = [], []
    for i, pose in zip(replay['order'], poses, strict=True):
        valid = np.isfinite(depth[i]) & (depth[i] > 0)
        points = rays[valid] * depth[i][valid,None]
        cloud.append(points @ pose[:3,:3].T + pose[:3,3])
        colors.append(rgb[i][valid])
    cloud, colors = np.concatenate(cloud), np.concatenate(colors)
    if args.ply_only:
        points = np.empty(len(cloud), dtype=[(k, '<f4') for k in 'xyz']
                          + [(k, 'u1') for k in ('red', 'green', 'blue')])
        for j, key in enumerate('xyz'):
            points[key] = cloud[:, j]
        for j, key in enumerate(('red', 'green', 'blue')):
            points[key] = colors[:, j]
        header = (f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n'
                  'property float x\nproperty float y\nproperty float z\n'
                  'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        with args.image.open('xb') as stream:
            stream.write(header.encode('ascii'))
            points.tofile(stream)
        return
    _, ids = np.unique(np.floor(cloud/.02).astype('int32'), axis=0, return_index=True)
    path = poses[:, :3, 3]
    span = np.ptp(path[:, [0,2]], axis=0)
    forward = [0,0,1] if span[1] >= span[0] else [1,0,0]
    fig, (left,right) = plt.subplots(1,2,figsize=(15,7.2))
    fig.patch.set_facecolor('#111820')
    draw(left,cloud[ids],colors[ids]/255,path,[.74,0,.67],[-.42,-1,.45],'Oblique view')
    draw(right,cloud[ids],colors[ids]/255,path,[forward[2],0,-forward[0]],forward,
         'Top view  |  camera path in cyan',True)
    fig.text(.035,.94,'OpenLORIS corridor  |  P32 registered RGB-D pointmap',
             color='white',fontsize=20,weight='bold')
    count = len(set(replay['order']))
    fig.text(.035,.895,f'{count}/{len(rgb)} keyframes registered  |  {len(cloud):,} depth samples',
             color='#aabccd',fontsize=12)
    fig.subplots_adjust(left=.03,right=.98,bottom=.02,top=.84,wspace=.05)
    with args.image.open('xb') as stream:
        fig.savefig(stream,format='png',dpi=140,facecolor=fig.get_facecolor())


if __name__ == '__main__':
    main()
