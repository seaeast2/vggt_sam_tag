"""Compare map lookup and full-image, patch ROI and place-first matching."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np

from matching import intrinsics, load_features, ranked_places, raster_indices, verify
from patch_map import sha, write_json
from spatial import PatchIndex, project_masks


def timing(values):
    return dict(p50_ms=float(np.median(values)), p95_ms=float(np.quantile(values, .95)))


def camera_with_error(camera, distance, angle):
    linear = camera[:, :3]
    center = -np.linalg.solve(linear, camera[:, 3])
    rotation = cv2.Rodrigues(np.deg2rad(angle) * np.array([0., 1., 0.]))[0]
    moved = linear @ rotation
    return np.column_stack((moved, -moved @ (center + [distance, 0., 0.])))


def load_map(path):
    meta = json.loads((path / 'map.json').read_text())
    if sha(Path(meta['source'])) != meta['input_sha256']:
        raise ValueError('Source is inconsistent with patch map')
    with np.load(meta['source']) as data:
        rgb, depth, cameras = (data[k] for k in ('rgb', 'depth', 'cameras'))
    with np.load(path / 'geometry.npz') as data:
        labels = data['labels']
        vertices, owner = data['vertices'], data['vertex_patch']
    points = {p['id']: vertices[owner == p['id']] for p in meta['patches']}
    points = {k: v[::max(1, len(v) // 6000)].astype(float) for k,v in points.items()}
    return meta, rgb, depth, cameras, labels, points


def group_footprints(meta):
    grouped = {}
    for observation in meta['observations']:
        grouped.setdefault(observation['patch_id'], []).append(observation)
    return grouped


def query_rois(index, points, camera, shape, margin=0, footprints=None):
    selected = index.query(camera, shape, margin)
    return project_masks(points, camera, shape, selected, margin=min(margin // 2, 8),
                         footprints=footprints)


def candidate_rois(meta, masks, labels, features, query, candidates):
    observed = [set(frame['patch_ids']) for frame in meta['frames']]
    matches = {}
    h, w = labels.shape[1:]
    qp = raster_indices(features[query], (h,w))
    for i in candidates:
        common = sorted(observed[i] & masks.keys())
        if not common:
            continue
        source_use = np.isin(labels[i][raster_indices(features[i], (h,w))], common)
        target_use = np.logical_or.reduce([masks[p][qp] > 0 for p in common])
        if source_use.any() and target_use.any():
            matches[i] = (source_use, target_use)
    return matches


def summarize(rows, positives):
    times = [r['matching_ms'] for r in rows]
    accepted = {(r['query'], r['candidate']) for r in rows if r['accepted']}
    tested = {(r['query'], r['candidate']) for r in rows}
    return dict(tested=len(rows), comparisons=sum(r['comparisons'] for r in rows),
                candidate_recall=len(tested & positives) / len(positives) if positives else None,
                accepted_recall=len(accepted & positives) / len(positives) if positives else None,
                accepted=len(accepted), matching=timing(times) if times else None)


def run(path, out):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(32)
    meta, _, depth, cameras, labels, points = load_map(path)
    n, h, w = depth.shape
    shape = (h,w)
    footprints = group_footprints(meta)
    indices, stats = {}, {}
    for name in ('linear', 'bvh', 'bsp'):
        count = 20
        start = perf_counter_ns()
        for _ in range(count):
            index = PatchIndex(meta['patches'], name)
        indices[name] = index
        stats[name] = dict(build_ms=(perf_counter_ns() - start) * 1e-6 / count, **index.stats())
    query_stats = {name: [] for name in indices}
    for _ in range(20):
        for camera in cameras:
            hits = []
            for name, index in indices.items():
                start = perf_counter_ns()
                hits.append(index.query(camera, shape))
                query_stats[name].append((perf_counter_ns() - start) * 1e-6)
            assert hits[0] == hits[1] == hits[2], 'Spatial indices disagree'
    for name, values in query_stats.items():
        stats[name]['query'] = timing(values)

    start = perf_counter_ns()
    features = load_features(path / 'features.npz')
    feature_load_ms = (perf_counter_ns() - start) * 1e-6
    ks = [intrinsics(p) for p in cameras]
    modes = {name: [] for name in ('full','patch','patch_fallback','place_patch','patch_drift_2',
                                   'patch_drift_5','patch_drift_10')}
    elapsed = {name: [] for name in modes}
    retrieval = []
    perturbation = {0: (0., 0, 0), 2: (.02, 2, 8), 5: (.05, 5, 16), 10: (.10, 10, 24)}
    unit = float(np.median(depth))
    for q in range(n):
        candidates = [i for i in range(n) if abs(i-q) >= 2]
        for mode, rows in modes.items():
            start = perf_counter_ns()
            selected = {i: (None, None) for i in candidates}
            if mode == 'place_patch':
                selected = {}
                for i in ranked_places(features, q, candidates)[:3]:
                    masks = query_rois(indices['bvh'], points, cameras[i], shape, 12, footprints)
                    selected.update(candidate_rois(meta, masks, labels, features, q, [i]))
            elif mode != 'full':
                drift = int(mode.rsplit('_', 1)[1]) if 'drift' in mode else 0
                distance, angle, margin = perturbation[drift]
                prior = camera_with_error(cameras[q], distance*unit, angle)
                masks = query_rois(indices['bvh'], points, prior, shape, margin, footprints)
                selected = candidate_rois(meta, masks, labels, features, q, candidates)
            tested = candidates if mode == 'patch_fallback' else selected
            for i in tested:
                source_use, target_use = selected.get(i, (None, None))
                pose, row = verify(features[i], features[q], depth[i], depth[q],
                                ks[i], ks[q], source_use, target_use)
                if mode == 'patch_fallback' and i in selected and pose is None:
                    _, full_row = verify(features[i], features[q], depth[i], depth[q], ks[i], ks[q])
                    full_row['matching_ms'] += row['matching_ms']
                    full_row['comparisons'] += row['comparisons']
                    row = full_row
                rows.append(dict(query=q, candidate=i, **row))
            elapsed[mode].append((perf_counter_ns() - start) * 1e-6)
            retrieval.append(dict(query=q, mode=mode, candidates=list(selected)))
    positives = {(r['query'],r['candidate']) for r in modes['full'] if r['accepted']}
    result = {name: dict(**summarize(rows, positives), end_to_end=timing(elapsed[name]))
              for name, rows in modes.items()}
    out.mkdir(parents=True)
    write_json(out / 'benchmark.json', dict(source=str(path), patches=len(meta['patches']),
               cpu_threads=cv2.getNumThreads(), feature_load_ms=feature_load_ms, index=stats,
               modes=result, baseline_verified_pairs=len(positives),
               positive_definition='verified original-depth baseline match; geometric proxy, not loop truth',
               perturbation=perturbation, details=modes, retrieval=retrieval))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.map, args.out)


if __name__ == '__main__':
    main()
