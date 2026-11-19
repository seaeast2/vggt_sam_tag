"""Compare strict RGB-D odometry spacing using the P32 feature verifier."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from matching import detect, verify
from patch_map import sha, write_json
from replay import bridge_pose, pose_metrics, reference, registered_depth, tracking_frames


def run(manifest, trajectory, calibration, out, tracking_manifest=None):
    meta = json.loads(manifest.read_text())
    frames = meta['frames']
    times = np.array([float(Path(row['rgb']).stem.removeprefix('rgb-')) for row in frames])
    rgb = []
    for row in frames:
        path = manifest.parent / row['rgb']
        if sha(path) != row['rgb_sha256']:
            raise ValueError(f'RGB hash mismatch: {path}')
        with Image.open(path) as image:
            rgb.append(np.asarray(image.convert('RGB').resize(tuple(meta['size_wh']), Image.Resampling.BICUBIC)))
    rgb = np.stack(rgb)
    depth, k = registered_depth(manifest, rgb.shape[1:3])
    cv2.setNumThreads(1)
    cv2.setRNGSeed(32)
    features = detect(rgb)
    intermediate = tracking_frames(tracking_manifest, rgb.shape[1:3], k) if tracking_manifest else []
    poses = [np.eye(4)]
    edges, failures, attempts, bridges = [], [], [], []
    for j in range(1,len(rgb)):
        for i in range(j-1,max(-1,j-4),-1):
            relative, stats = verify(features[i],features[j],depth[i],depth[j],k,k)
            attempts.append(dict(source=i,target=j,**stats))
            if relative is not None:
                poses.append(poses[i] @ np.linalg.inv(relative))
                edges.append(dict(source=i,target=j))
                break
        else:
            bridge = bridge_pose(j-1, j, times, features, depth, k, intermediate) if intermediate else None
            if bridge is None:
                failures.append(dict(position=j,image=j))
                break
            relative, steps = bridge
            poses.append(poses[-1] @ np.linalg.inv(relative))
            edges.append(dict(source=j-1,target=j))
            bridges.append(dict(source=j-1,target=j,intermediate_timestamps=steps[:-1]))
    # Continue pair diagnostics past a global tracking loss without injecting poses.
    cv2.setRNGSeed(32)
    adjacent = []
    for i in range(len(rgb)-1):
        _, stats = verify(features[i],features[i+1],depth[i],depth[i+1],k,k)
        adjacent.append(dict(source=i,target=i+1,time_s=float(times[i+1]-times[0]),**stats))
    truth = reference(trajectory,calibration,times.tolist())
    poses = np.stack(poses)
    n = len(poses)
    out.mkdir(parents=True)
    np.save(out/'full_poses.npy',poses)
    np.save(out/'reference.npy',truth[:n])
    np.savez_compressed(out/'rgb_input.npz',rgb=rgb)
    write_json(out/'replay.json',dict(rgb_input=str(out/'rgb_input.npz'),manifest=str(manifest),
               order=list(range(n)),input_count=len(rgb),losses=failures,odometry_edges=len(edges),
               edges=edges,attempts=attempts,adjacent=adjacent,timestamps=times.tolist(),
               tracking_manifest=str(tracking_manifest) if tracking_manifest else None,
               tracking_bridges=bridges,
               reference=dict(trajectory=str(trajectory),calibration=str(calibration),
                              trajectory_sha256=sha(trajectory),calibration_sha256=sha(calibration)),
               baseline=pose_metrics(poses,truth[:n])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('manifest','trajectory','calibration','out'):
        p.add_argument(arg,type=Path)
    p.add_argument('--tracking-manifest',type=Path)
    a=p.parse_args()
    run(a.manifest,a.trajectory,a.calibration,a.out,a.tracking_manifest)
