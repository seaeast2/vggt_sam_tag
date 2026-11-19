"""Replay RGB-D odometry with full-image or patch-restricted loop verification."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from benchmark import candidate_rois, group_footprints, load_map, query_rois, timing
from matching import detect, load_features, ranked_places, transform, verify
from patch_map import sha, write_json
from spatial import PatchIndex


def registered_depth(path, shape):
    manifest = json.loads(path.read_text())
    w,h = manifest['size_wh']
    if (h,w) != shape:
        raise ValueError('RGB-D and patch raster differ')
    depths = []
    for row in manifest['frames']:
        dep = path.parent / row['depth']
        if sha(dep) != row['depth_sha256']:
            raise ValueError(f'Sensor hash mismatch: {dep}')
        with Image.open(dep) as image:
            depths.append(np.asarray(image.resize((w,h), Image.Resampling.NEAREST)) /
                          manifest['units_per_meter'])
    fx,fy,cx,cy = manifest['color_k_fx_fy_cx_cy']
    w0,h0 = manifest['source_size_wh']
    k = np.array([[fx*w/w0, 0., (cx+.5)*w/w0-.5],
                  [0., fy*h/h0, (cy+.5)*h/h0-.5], [0.,0.,1.]])
    return np.stack(depths), k


def reference(trajectory, calibration, timestamps):
    values = np.loadtxt(trajectory)
    if min(timestamps) < values[0,0] - .025 or max(timestamps) > values[-1,0] + .025:
        raise ValueError('Reference does not cover RGB timestamps')
    timestamps = np.clip(timestamps, values[0,0], values[-1,0])
    store = cv2.FileStorage(str(calibration), cv2.FILE_STORAGE_READ)
    nodes = store.getNode('trans_matrix')
    camera = None
    for i in range(nodes.size()):
        row = nodes.at(i)
        if (row.getNode('parent_frame').string(), row.getNode('child_frame').string()) == (
                'base_link', 'd400_color_optical_frame'):
            camera = row.getNode('matrix').mat()
            break
    store.release()
    if camera is None:
        raise ValueError('Missing calibrated camera optical transform')
    rotations = Slerp(values[:,0], Rotation.from_quat(values[:,4:8]))(timestamps).as_matrix()
    translations = np.stack([np.interp(timestamps, values[:,0], a)
                             for a in values[:,1:4].T], axis=1)
    return np.stack([transform(r,t) @ camera for r,t in zip(rotations,translations,strict=True)])


def delta(estimate):
    return np.r_[estimate[:3,3], Rotation.from_matrix(estimate[:3,:3]).as_rotvec()]


def decode(numbers):
    return transform(Rotation.from_rotvec(numbers[3:]).as_matrix(), numbers[:3])


def optimize(poses, edges):
    if len(poses) < 2:
        return poses.copy()
    first = poses[0]

    def unpack(values):
        return np.array([first, *(decode(x) for x in values.reshape(-1,6))])

    constraints = [(i,j,np.linalg.inv(relative)) for i,j,relative in edges]

    def residual(values):
        estimates = unpack(values)
        return np.concatenate([delta(inverse @ np.linalg.inv(estimates[j]) @ estimates[i])
                               for i,j,inverse in constraints])

    start = np.concatenate([delta(p) for p in poses[1:]])
    solution = least_squares(residual, start, loss='huber', f_scale=.05, max_nfev=60)
    return unpack(solution.x)


def pose_metrics(estimate, truth):
    if len(estimate) < 2:
        return None
    a, b = estimate[:, :3,3], truth[:, :3,3]
    aa, bb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-aa).T @ (b-bb))
    handedness = np.linalg.det(u @ vt)
    rotation = u @ np.diag([1,1,handedness]) @ vt
    aligned = (a-aa) @ rotation + bb
    trans, rot = [], []
    for i in range(1,len(estimate)):
        actual = np.linalg.inv(truth[i]) @ truth[i-1]
        measured = np.linalg.inv(estimate[i]) @ estimate[i-1]
        error = delta(np.linalg.inv(actual) @ measured)
        trans.append(np.linalg.norm(error[:3]))
        rot.append(np.degrees(np.linalg.norm(error[3:])))
    return dict(ate_rmse_m=float(np.sqrt(np.mean(np.sum((aligned-b)**2,axis=1)))),
                rpe_rmse_m=float(np.sqrt(np.mean(np.square(trans)))),
                rpe_rotation_rmse_deg=float(np.sqrt(np.mean(np.square(rot)))))


def valid_loop(truth, i, j):
    return near_loop(np.linalg.inv(truth[j]) @ truth[i])


def near_loop(pose):
    error = delta(pose)
    return np.linalg.norm(error[:3]) <= .8 and np.linalg.norm(error[3:]) <= np.deg2rad(35)


def tracking_frames(path, shape, k):
    depth, bridge_k = registered_depth(path, shape)
    np.testing.assert_allclose(k, bridge_k)
    manifest = json.loads(path.read_text())
    images, timestamps = [], []
    for row in manifest['frames']:
        image_path = path.parent / row['rgb']
        if sha(image_path) != row['rgb_sha256']:
            raise ValueError(f'Bridge image hash mismatch: {image_path}')
        with Image.open(image_path) as image:
            images.append(np.asarray(image.convert('RGB').resize(
                (shape[1], shape[0]), Image.Resampling.BICUBIC)))
        timestamps.append(float(Path(row['rgb_archive_path']).stem))
    return list(zip(timestamps, detect(images), depth, strict=True))


def bridge_pose(a, b, times, features, depth, k, intermediate):
    start, end = sorted((times[a], times[b]))
    inside = [row for row in intermediate if start < row[0] < end]
    if not inside:
        return None
    if times[b] < times[a]:
        inside.reverse()
    chain = [(times[a], features[a], depth[a]), *inside, (times[b], features[b], depth[b])]
    reached = {0: (np.eye(4), [])}
    for j, (timestamp, target, target_depth) in enumerate(chain[1:], 1):
        for i in range(j - 1, -1, -1):
            if i not in reached:
                continue
            _, source, source_depth = chain[i]
            pose, _ = verify(source, target, source_depth, target_depth, k, k)
            if pose is not None:
                previous, steps = reached[i]
                reached[j] = (pose @ previous, steps + [timestamp])
                break
    return reached.get(len(chain) - 1)


def evaluate(map_path, out, trajectory, calibration, controlled, tracking_manifest=None):
    cv2.setRNGSeed(32)
    cv2.setNumThreads(1)
    meta, rgb, _, cameras, labels, _ = load_map(map_path)
    shape = rgb.shape[1:3]
    depth,k = registered_depth(Path(meta['manifest']), shape)
    timestamps = [fr['timestamp'] for fr in meta['frames']]
    truth = reference(trajectory,calibration,timestamps)
    features = load_features(map_path / 'features.npz')
    intermediate = tracking_frames(tracking_manifest, shape, k) if tracking_manifest else []
    index = PatchIndex(meta['patches'])
    footprints = group_footprints(meta)
    with np.load(map_path / 'geometry.npz') as mesh:
        vertex_frames = mesh['frame_v_u'][:, 0]
        vertex_patches = mesh['vertex_patch']
        vertices = mesh['vertices']
    by_frame = []
    for frame in meta['frames']:
        i = frame['frame']
        visible = vertex_frames == i
        p, v = vertex_patches[visible], vertices[visible]
        samples = {patch: v[p == patch].astype(float) for patch in frame['patch_ids']}
        rings = {patch: [obs for obs in footprints[patch] if obs['frame'] == i]
                 for patch in frame['patch_ids']}
        by_frame.append((samples, rings))
    order = list(range(len(rgb)))
    if controlled:
        # Revisit the same frames using an exactly reproducible reverse playback.
        order += order[-2::-1]
    poses = [np.eye(4)]
    odometry, losses, bridges = [], [], []
    for j in range(1,len(order)):
        b = order[j]
        found = None
        for i in range(j-1,max(-1,j-4),-1):
            a = order[i]
            pose, row = verify(features[a],features[b],depth[a],depth[b],k,k)
            if pose is not None:
                found = i,pose
                break
        if found is None and intermediate:
            bridge = bridge_pose(order[j-1], b, timestamps, features, depth, k, intermediate)
            if bridge is not None:
                pose, steps = bridge
                found = j-1, pose
                bridges.append(dict(source=j-1, target=j, intermediate_timestamps=steps[:-1]))
        if found is None:
            losses.append(dict(position=j, image=b))
            break
        i,pose = found
        poses.append(poses[i] @ np.linalg.inv(pose))
        odometry.append((i,j,pose))
    order = order[:len(poses)]
    expected = {(i,j) for j in range(len(order)) for i in range(j-4)
                if valid_loop(truth,order[i],order[j])}
    out.mkdir(parents=True)
    modes = {}
    for mode in ('full','patch','patch_fallback'):
        eligible = tested = 0
        options, chosen, elapsed = [], [], []
        for j,b in enumerate(order):
            start = perf_counter_ns()
            eligible_positions = list(range(j-4))
            unique = {order[i]: i for i in eligible_positions}
            eligible += len(unique)
            prioritized = ranked_places(features,b,unique)[:4]
            accepted = []
            for a in prioritized:
                i = unique[a]
                rg = {}
                if mode != 'full':
                    points, rings = by_frame[a]
                    pixels = query_rois(index,points,cameras[a],shape,12,rings)
                    rg = candidate_rois(meta,pixels,labels,features,b,[a])
                if mode == 'patch' and a not in rg:
                    continue
                source_use,target_use = rg.get(a,(None,None))
                pose,row = verify(features[a],features[b],depth[a],depth[b],k,k,
                                  source_use,target_use)
                if mode == 'patch_fallback' and a in rg and (pose is None or not near_loop(pose)):
                    pose, retry = verify(features[a],features[b],depth[a],depth[b],k,k)
                    retry['matching_ms'] += row['matching_ms']
                    retry['comparisons'] += row['comparisons']
                    row = retry
                tested += 1
                options.append(dict(source=i,target=j,**row))
                if pose is None or not near_loop(pose):
                    continue
                accepted.append((row['inliers'],i,pose))
            # Require a second candidate to agree when two swaths of the same place exist.
            if accepted:
                accepted.sort(key=lambda a:a[0],reverse=True)
                _,i,pose = accepted[0]
                predicted = poses[i] @ np.linalg.inv(pose)
                peers = [poses[x] @ np.linalg.inv(p) for _,x,p in accepted[1:]]
                if not peers or any(np.linalg.norm(delta(np.linalg.inv(predicted) @ peer)[:3]) <= .3
                                    for peer in peers):
                    chosen.append((i,j,pose))
            elapsed.append((perf_counter_ns()-start)*1e-6)
        start = perf_counter_ns()
        refined = optimize(np.stack(poses), odometry+chosen) if chosen else np.stack(poses)
        graph_ms = (perf_counter_ns()-start)*1e-6
        hits = {(i,j) for i,j,_ in chosen}
        correct = hits & expected
        targets = {j for _,j in expected}
        modes[mode] = dict(eligible=eligible,tested=tested,query_timing=timing(elapsed),
                           graph_ms=graph_ms, expected_pairs=len(expected),
                           precision=len(correct)/len(hits) if hits else None,
                           revisit_query_recall=len({j for _,j in correct})/len(targets) if targets else None,
                           edges=[dict(source=i,target=j,correct=(i,j) in expected) for i,j in sorted(hits)],
                           pose=pose_metrics(refined,truth[order]),details=options)
        np.save(out / f'{mode}_poses.npy', refined)
    np.save(out / 'reference.npy',truth[order])
    write_json(out / 'replay.json',dict(source=str(map_path), controlled_reverse_playback=controlled,
               order=order, odometry_edges=len(odometry), losses=losses,
               tracking_manifest=str(tracking_manifest) if tracking_manifest else None,
               tracking_bridges=bridges,
               reference=dict(trajectory=str(trajectory),trajectory_sha256=sha(trajectory),
                              calibration=str(calibration),calibration_sha256=sha(calibration)),
               baseline=pose_metrics(np.stack(poses),truth[order]),modes=modes))
    print(json.dumps({m:{k:v for k,v in d.items() if k!='details'} for m,d in modes.items()},indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('map',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('trajectory',type=Path)
    p.add_argument('calibration',type=Path)
    p.add_argument('--controlled',action='store_true')
    p.add_argument('--tracking-manifest', type=Path)
    args = p.parse_args()
    evaluate(args.map,args.out,args.trajectory,args.calibration,args.controlled,args.tracking_manifest)


if __name__ == '__main__':
    main()
