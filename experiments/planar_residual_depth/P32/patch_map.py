"""Build a versioned patch map from P31 observations and original camera rasters."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from matching import detect, raster_indices


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def projected_overlap(points, camera, labels, plane):
    h, w = labels.shape
    uvz = points @ camera[:, :3].T + camera[:, 3]
    uvz = uvz[uvz[:, 2] > 0]
    uv = np.rint(uvz[:, :2] / uvz[:, 2:]).astype(int)
    u, v = uv.T
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    hits = np.count_nonzero(labels[v[inside], u[inside]] == plane)
    return hits, hits / max(len(points), 1)


def cluster(planes, samples, cameras, labels, support, distance):
    compatible, scores = set(), []
    for i, a in planes.items():
        for j, b in planes.items():
            if j <= i or a['frame'] == b['frame']:
                continue
            similarity = np.dot(a['normal'], b['normal'])
            if abs(similarity) < np.cos(np.deg2rad(8)):
                continue
            if abs(a['offset'] - np.copysign(1, similarity) * b['offset']) > distance:
                continue
            def check(src, dst):
                frame = planes[dst]['frame']
                ids = support.get(f'{src}_support_{frame}', [])
                if len(ids) == 0:
                    return 0, 0.
                hits = np.count_nonzero(labels[frame].ravel()[ids] == dst)
                return hits, hits / len(ids)
            evidence = max(check(i, j), check(j, i))
            if evidence[0] < 18 or evidence[1] < .35:
                continue
            ai, ar = projected_overlap(samples[i], cameras[b['frame']], labels[b['frame']], j)
            bi, br = projected_overlap(samples[j], cameras[a['frame']], labels[a['frame']], i)
            if min(ai, bi) < 15 or min(ar, br) < .15:
                continue
            compatible.add((i, j))
            scores.append((min(ar, br), i, j))
    groups = {i: {i} for i in planes}
    owner = {i: i for i in planes}
    history = []
    for score, i, j in sorted(scores, reverse=True):
        a, b = sorted((owner[i], owner[j]))
        if a == b:
            continue
        if all(tuple(sorted((x, y))) in compatible for x in groups[a] for y in groups[b]):
            history.append(dict(retained=a, merged=b, supporting_pair=[i, j], overlap=score))
            for observation in groups[b]:
                owner[observation] = a
            groups[a].update(groups.pop(b))
    return groups, owner, history


def footprints(labels, plane_id, camera, normal, offset):
    mask = (labels == plane_id).astype('uint8')
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = -np.linalg.solve(camera[:, :3], camera[:, 3])
    rays_from_pixels = np.linalg.inv(camera[:, :3]).T
    rings = []
    for index, contour in enumerate(contours):
        uv = contour[:, 0]
        parent = int(hierarchy[0, index, 3])
        rays = np.column_stack((uv, np.ones(len(uv)))) @ rays_from_pixels
        depth = (offset - center @ normal) / (rays @ normal)
        rings.append(dict(parent=parent, uv=uv.tolist(), xyz=(center + rays * depth[:, None]).tolist()))
    return rings


def metadata(manifest, n, h, w):
    if manifest is None:
        return [dict(frame=i, image=None, timestamp=None, size_wh=[w, h]) for i in range(n)]
    source = json.loads(manifest.read_text())
    if source['size_wh'] != [w, h] or len(source['frames']) != n:
        raise ValueError('RGB-D input manifest does not match P31 raster')
    frames = []
    for i, f in enumerate(source['frames']):
        image = manifest.parent / f['rgb']
        if sha(image) != f['rgb_sha256']:
            raise ValueError(f'Image hash mismatch for {image}')
        stem = Path(f.get('rgb_archive_path', '')).stem
        frames.append(dict(frame=i, image=str(image), source_index=f['index'],
                           timestamp=float(stem) if stem else None, size_wh=[w, h],
                           source_size_wh=source['source_size_wh']))
    return frames


def export_mesh(path, vertices, colors, faces, observation_ids, patch_ids):
    records = np.empty(len(vertices), dtype=[
        (name, '<f4') for name in 'xyz'] + [(name, 'u1') for name in ('red', 'green', 'blue')]
        + [('observation_id', '<i4'), ('patch_id', '<i4')])
    for name, values in zip(('x', 'y', 'z', 'red', 'green', 'blue'),
                            (*vertices.T, *colors.T), strict=True):
        records[name] = values
    records['observation_id'], records['patch_id'] = observation_ids, patch_ids
    triangles = np.empty(len(faces), dtype=[('n', 'u1'), ('indices', '<i4', (3,))])
    triangles['n'], triangles['indices'] = 3, faces
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {len(vertices)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'property int observation_id\nproperty int patch_id\n'
        f'element face {len(faces)}\n'
        'property list uchar int vertex_indices\nend_header\n'
    )
    with path.open('xb') as stream:
        stream.write(header.encode())
        records.tofile(stream)
        triangles.tofile(stream)


def build(run, out, manifest=None):
    summary = json.loads((run / 'summary.json').read_text())
    planes = {p['id']: p for p in summary['planes']}
    source = Path(summary['source'])
    if sha(source) != summary['source_sha256']:
        raise ValueError('P31 input hash mismatch')
    with np.load(source) as z:
        cameras = z['cameras']
        raw = z['depth']
        rgb = z['rgb']
    with np.load(run / 'pointmaps.npz') as z:
        labels, xyz = z['labels'], z['xyz']
    with np.load(run / 'mesh.npz') as z:
        vertices, faces, observation_id, pixels = (z[k] for k in
            ('vertices', 'faces', 'plane_id', 'frame_v_u'))
    with np.load(run / 'observations.npz') as z:
        support = dict(z)
    if not np.array_equal(observation_id, labels[tuple(pixels.T)]) or not np.array_equal(
            vertices, xyz[tuple(pixels.T)]):
        raise ValueError('Mesh does not match the original P31 pixel mapping')
    if not np.array_equal(observation_id[faces], np.repeat(observation_id[faces[:, :1]], 3, axis=1)):
        raise ValueError('Triangle crosses observation patches')
    samples = {i: vertices[observation_id == i] for i in planes}
    if any(len(v) == 0 for v in samples.values()):
        raise ValueError('Plane without mesh vertices')
    subset = {i: s[::max(1, len(s) // 400)] for i, s in samples.items()}
    groups, owner, history = cluster(planes, subset, cameras, labels, support,
                                     .015 * float(np.median(raw)))
    n, h, w = labels.shape
    observations = []
    for i, plane in sorted(planes.items()):
        frame = plane['frame']
        ys, xs = np.nonzero(labels[frame] == i)
        observations.append(dict(id=i, patch_id=owner[i], frame=frame,
                         plane=[*plane['normal'], plane['offset']],
                         roi_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                         pixels=len(xs), support_frames=plane['support_frames'],
                         rings=footprints(labels[frame], i, cameras[frame],
                                          np.asarray(plane['normal']), plane['offset'])))
    near = {root: set() for root in groups}
    # Detect adjacent image footprints within the one-pixel erosion moat on each side.
    lut = np.zeros(max(planes, default=0) + 1, dtype=np.int32)
    for i, root in owner.items():
        lut[i] = root
    mapped = lut[labels]
    for d in (1, 2, 3, 4):
        pairs = [(mapped[:, :, :-d], mapped[:, :, d:]), (mapped[:, :-d], mapped[:, d:])]
        for a, b in pairs:
            mask = (a > 0) & (b > 0) & (a != b)
            for i, j in np.unique(np.column_stack((a[mask], b[mask])), axis=0):
                near[int(i)].add(int(j))
                near[int(j)].add(int(i))
    patches = []
    for root, ids in sorted(groups.items()):
        points = np.concatenate([samples[i] for i in sorted(ids)])
        exemplar = max(sorted(ids), key=lambda i: len(samples[i]))
        p = planes[exemplar]
        patches.append(dict(id=root, submap=run.name, version=1, observations=sorted(ids),
                            exemplar=exemplar, plane=[*p['normal'], p['offset']],
                            bounds=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                            neighbor_candidates=sorted(near[root])))
    frames = metadata(manifest, n, h, w)
    with np.load(source) as z:
        confidence = z['confidence']
    features = detect(rgb)
    feature_offsets = np.r_[0, np.cumsum([len(f.uv) for f in features])]
    feature_owners = [labels[i][raster_indices(f, (h,w))] for i, f in enumerate(features)]
    for frame in frames:
        i = frame['frame']
        frame['patch_ids'] = np.unique(mapped[i][mapped[i] > 0]).tolist()
        frame['projection'] = cameras[i].tolist()
        k, rotation, center, *_ = cv2.decomposeProjectionMatrix(cameras[i])
        camera_to_submap = np.eye(4)
        camera_to_submap[:3, :3] = rotation.T
        camera_to_submap[:3, 3] = (center[:3] / center[3]).ravel()
        frame['intrinsics'] = (k / k[2, 2]).tolist()
        frame['camera_to_submap'] = camera_to_submap.tolist()
        frame['feature_range'] = feature_offsets[i:i+2].tolist()
    assert confidence.shape == labels.shape and rgb.shape[:3] == labels.shape
    out.mkdir(parents=True)
    hashes = {name: sha(run / name) for name in
              ('summary.json', 'mesh.npz', 'observations.npz', 'pointmaps.npz')}
    write_json(out / 'map.json', dict(schema_version=1, coordinate_system='P31 source submap',
               submap=run.name, submap_transform_version=1, submap_to_map=np.eye(4).tolist(),
               source=str(source), input_sha256=summary['source_sha256'],
               input_run=str(run), inputs=hashes, manifest=str(manifest) if manifest else None,
               raster=[n,h,w], frames=frames, patches=patches, observations=observations,
               merge_history=history))
    np.savez_compressed(out / 'geometry.npz', vertices=vertices, faces=faces,
                        vertex_observation=observation_id, vertex_patch=lut[observation_id],
                        frame_v_u=pixels, rgb=rgb[tuple(pixels.T)],
                        confidence=confidence[tuple(pixels.T)], labels=mapped,
                        observation_labels=labels)
    np.savez_compressed(out / 'observation_evidence.npz', **support)
    feature_observation = np.concatenate(feature_owners)
    np.savez_compressed(out / 'features.npz', offsets=feature_offsets,
                        uv=np.concatenate([f.uv for f in features]),
                        descriptors=np.concatenate([f.descriptors for f in features]),
                        signatures=np.stack([f.signature for f in features]),
                        observation_id=feature_observation, patch_id=lut[feature_observation])
    export_mesh(out / 'mesh.ply', vertices, rgb[tuple(pixels.T)], faces,
                observation_id, lut[observation_id])
    with np.load(out / 'geometry.npz') as saved:
        for name, original in [('vertices',vertices), ('faces',faces), ('frame_v_u',pixels),
                               ('observation_labels',labels)]:
            np.testing.assert_array_equal(saved[name], original)
    print(f'{len(observations)} observations, {len(patches)} map patches, {len(history)} merges')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    build(args.run, args.out, args.manifest)


if __name__ == '__main__':
    main()
