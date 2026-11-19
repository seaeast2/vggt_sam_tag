"""P31a CPU pilot using the frozen P26 visibility mask and P27 raster cameras."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


LIMITS = dict(tile=32, min_fit_points=80, min_region=300, min_support=40,
              normal_deg=15, fit_residual=.003, region_residual=.012,
              support_residual=.006, min_support_fraction=.65, max_move=.03,
              min_baseline=.02, max_rgb_distance=60, max_roundtrip_px=2.5)
COS_NORMAL = np.cos(np.deg2rad(LIMITS['normal_deg']))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def normals(xyz, keep):
    dx = np.roll(xyz, -1, axis=1) - np.roll(xyz, 1, axis=1)
    dy = np.roll(xyz, -1, axis=0) - np.roll(xyz, 1, axis=0)
    cross = np.cross(dx, dy)
    lengths = np.linalg.norm(cross, axis=2)
    valid = cv2.erode(keep.astype('uint8'), np.ones((3, 3), 'uint8'),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    unit = cross / np.maximum(lengths[..., None], 1e-12)
    unit[~valid] = 0
    return unit


def fit(points):
    center = np.median(points, axis=0)
    delta = points - center
    values, vectors = np.linalg.eigh(delta.T @ delta)
    if values[1] <= 1e-10:
        return None
    normal = vectors[:, 0]
    return normal, float(np.median(points @ normal))


def modes(xyz, depth, unit, high, spatial, keep):
    step = LIMITS['tile']
    h, w = depth.shape
    seeds = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            sl = np.s_[y:y + step, x:x + step]
            use = high[sl] & ~spatial[sl] & keep[sl]
            row, col = np.nonzero(use)
            if len(row) < LIMITS['min_fit_points']:
                continue
            ids = (row + y) * w + col + x
            points = xyz[sl][use].astype(float)
            plane = fit(points)
            if plane is None:
                continue
            normal, offset = plane
            distances = abs(points @ normal - offset) / depth[sl][use]
            inlier = (distances <= LIMITS['fit_residual']) & (
                abs(unit[sl][use] @ normal) >= COS_NORMAL)
            if inlier.sum() < LIMITS['min_fit_points']:
                continue
            plane = fit(points[inlier])
            if plane is not None:
                seeds.append((int(inlier.sum()), ids[inlier], plane))
    return sorted(seeds, key=lambda seed: seed[0], reverse=True)


def check_support(i, ids, plane, xyz, depth, unit, keep, rgb, cameras, centers):
    normal, offset = plane
    h, w = depth.shape[1:]
    y, x = np.divmod(ids, w)
    source = xyz[i, y, x]
    support = []
    # Opposite parity views supply validation and never fit this candidate.
    for j in range(1 - i % 2, len(depth), 2):
        if np.linalg.norm(centers[i] - centers[j]) / np.median(depth[i, y, x]) < LIMITS['min_baseline']:
            continue
        q = source @ cameras[j, :, :3].T + cameras[j, :, 3]
        front = q[:, 2] > 0
        uv = np.rint(q[front, :2] / q[front, 2:]).astype(int)
        u, v = uv.T
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v = u[inside], v[inside]
        original = np.flatnonzero(front)[inside]
        close = abs(depth[j, v, u] - q[original, 2]) <= .025 * q[original, 2]
        visible = keep[j, v, u] & close
        if visible.sum() < LIMITS['min_support']:
            continue
        u, v, original = u[visible], v[visible], original[visible]
        peer = xyz[j, v, u]
        back = peer @ cameras[i, :, :3].T + cameras[i, :, 3]
        back_uv = back[:, :2] / np.maximum(back[:, 2:], 1e-12)
        source_uv = np.column_stack((x[original], y[original]))
        roundtrip = np.linalg.norm(back_uv - source_uv, axis=1) <= LIMITS['max_roundtrip_px']
        color = np.linalg.norm(rgb[i, y[original], x[original]].astype(float) -
                               rgb[j, v, u], axis=1)
        agree = ((abs(peer @ normal - offset) <= LIMITS['support_residual'] * depth[j, v, u])
                 & (abs(unit[j, v, u] @ normal) >= COS_NORMAL)
                 & roundtrip & (color <= LIMITS['max_rgb_distance']))
        if agree.mean() >= LIMITS['min_support_fraction']:
            support.append((j, np.unique((v * w + u)[agree])))
    return support


def refine(rgb, xyz, confidence, keep, cameras, depth):
    n, h, w = depth.shape
    y, x = np.indices((h, w))
    spatial = (x // 16 + 3 * (y // 16)) % 5 == 1
    high = np.stack([c >= np.median(c[k]) for c, k in zip(confidence, keep, strict=True)])
    unit = np.stack([normals(p, k) for p, k in zip(xyz, keep, strict=True)])
    centers = np.linalg.solve(cameras[:, :, :3], -cameras[:, :, 3])
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    corrected = depth.copy()
    labels = np.zeros_like(depth, dtype='int32')
    points = xyz.copy()
    records, audits, evidence = [], [], {}
    for i in range(n):
        rays = pixels @ np.linalg.inv(cameras[i, :, :3]).T
        unclaimed = keep[i].copy()
        counts = dict(frame=i, seeded=0, insufficient_region=0, insufficient_support=0,
                      overlapping_seed=0, accepted=0)
        for _, seed, plane in modes(xyz[i], depth[i], unit[i], high[i], spatial, keep[i]):
            counts['seeded'] += 1
            if unclaimed.ravel()[seed].mean() < .5:
                counts['overlapping_seed'] += 1
                continue
            normal, offset = plane
            residual = abs(xyz[i] @ normal - offset) / depth[i]
            compatible = unclaimed & (residual <= LIMITS['region_residual']) & (
                abs(unit[i] @ normal) >= COS_NORMAL)
            _, components = cv2.connectedComponents(compatible.astype('uint8'), connectivity=4)
            votes = np.bincount(components.ravel()[seed])
            if len(votes) <= 1:
                counts['insufficient_region'] += 1
                continue
            region = components == (1 + votes[1:].argmax())
            region = cv2.erode(region.astype('uint8'), np.ones((3, 3), 'uint8'),
                               borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            denom = rays @ normal
            proposal = np.divide(offset - centers[i] @ normal, denom,
                                 out=np.zeros_like(denom), where=abs(denom) > 1e-8)
            region &= (proposal > 0) & (abs(proposal - depth[i]) <= LIMITS['max_move'] * depth[i])
            ids = np.flatnonzero(region)
            if len(ids) < LIMITS['min_region']:
                counts['insufficient_region'] += 1
                continue
            spatial_sample = ids[spatial.ravel()[ids]][::3]
            support = check_support(i, spatial_sample, plane, xyz, depth, unit, keep,
                                    rgb, cameras, centers)
            if not support:
                counts['insufficient_support'] += 1
                continue
            plane_id = len(records) + 1
            labels[i, region] = plane_id
            corrected[i, region] = proposal[region]
            points[i, region] = centers[i] + rays[region] * corrected[i, region, None]
            np.testing.assert_allclose(points[i, region] @ normal, offset, atol=2e-6)
            unclaimed[region] = False
            counts['accepted'] += 1
            evidence[f'{plane_id}_fit'] = seed
            evidence[f'{plane_id}_source_holdout'] = spatial_sample
            for j, indices in support:
                evidence[f'{plane_id}_support_{j}'] = indices
            records.append(dict(id=plane_id, frame=i, normal=normal.tolist(), offset=offset,
                                support_frames={j: len(k) for j, k in support},
                                fit_pixels=len(seed), assigned_pixels=len(ids)))
        audits.append(counts)
        print(f'frame {i:02d} {counts}', flush=True)
    return corrected, points, labels, high, spatial, records, audits, evidence


def export_ply(path, xyz, rgb, use, changed=None):
    vertices = np.empty(np.count_nonzero(use), dtype=[
        (key, '<f4') for key in 'xyz'] + [(key, 'u1') for key in ('red', 'green', 'blue')]
        + [(key, '<u2') for key in ('frame_id', 'pixel_u', 'pixel_v')])
    color = rgb[use].copy()
    if changed is not None:
        color[changed[use]] = (255, 145, 20)
    for key, values in zip('xyz', xyz[use].T, strict=True):
        vertices[key] = values
    for key, values in zip(('red', 'green', 'blue'), color.T, strict=True):
        vertices[key] = values
    for key, values in zip(('frame_id', 'pixel_v', 'pixel_u'), np.nonzero(use), strict=True):
        vertices[key] = values
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(vertices)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'property ushort frame_id\nproperty ushort pixel_u\nproperty ushort pixel_v\nend_header\n')
    with path.open('xb') as stream:
        stream.write(header.encode())
        vertices.tofile(stream)


def export_faces(path, points, labels):
    # Mesh finite visible footprints on the raster, preserving holes and patch boundaries.
    mask = labels > 0
    indices = np.full(labels.shape, -1, dtype='int32')
    indices[mask] = np.arange(mask.sum())
    vertices = points[mask]
    a, b, c, d = labels[:, :-1, :-1], labels[:, :-1, 1:], labels[:, 1:, :-1], labels[:, 1:, 1:]
    ai, bi, ci, di = (indices[:, :-1, :-1], indices[:, :-1, 1:],
                      indices[:, 1:, :-1], indices[:, 1:, 1:])
    first = (a > 0) & (a == b) & (a == c)
    second = (d > 0) & (d == c) & (d == b)
    faces = np.concatenate((np.stack((ai[first], bi[first], ci[first]), axis=1),
                            np.stack((di[second], ci[second], bi[second]), axis=1)))
    np.savez_compressed(path, vertices=vertices, faces=faces, plane_id=labels[mask],
                        frame_v_u=np.stack(np.nonzero(mask), axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='P27 input.npz')
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    with np.load(args.source) as data:
        rgb, xyz, confidence, keep, cameras, depth = (
            data[k] for k in ('rgb', 'xyz', 'confidence', 'keep', 'cameras', 'depth'))
    corrected, points, labels, high, spatial, planes, audits, evidence = refine(
        rgb, xyz, confidence, keep, cameras, depth)
    assigned = labels > 0
    changed = assigned & (corrected != depth)
    low = changed & ~high
    assert not np.any(assigned & ~keep)
    assert np.array_equal(points[~assigned], xyz[~assigned])
    np.savez_compressed(args.out / 'pointmaps.npz', raw=depth, all_confidence=corrected,
                        low_only=np.where(low, corrected, depth), xyz=points,
                        confidence=confidence, keep=keep, labels=labels, high=high,
                        spatial_holdout=spatial, changed=changed, delta=corrected - depth)
    np.savez_compressed(args.out / 'observations.npz', **evidence)
    export_faces(args.out / 'mesh.npz', points, labels)
    export_ply(args.out / 'baseline.ply', xyz, rgb, keep)
    export_ply(args.out / 'low_only.ply', np.where(low[..., None], points, xyz), rgb, keep, low)
    export_ply(args.out / 'all_confidence.ply', points, rgb, keep, changed)
    n, h, w = depth.shape
    sheet = Image.new('RGB', (4 * w, ((n + 3) // 4) * h))
    palette = np.random.default_rng(31).integers(40, 256, (len(planes) + 1, 3))
    for i, image in enumerate(rgb):
        overlay = np.where(assigned[i, ..., None], .45 * image + .55 * palette[labels[i]], image)
        sheet.paste(Image.fromarray(overlay.astype('uint8')), (i % 4 * w, i // 4 * h))
    sheet.save(args.out / 'regions.jpg', quality=93)
    with args.source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    write_json(args.out / 'summary.json', dict(
        status='CPU_PILOT' if planes else 'NO_ACCEPTED_PLANES', limits=LIMITS, source=str(args.source),
        source_sha256=digest, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
        frames=n, input_pixels=int(depth.size), visible_pixels=int(keep.sum()),
        changed_pixels=int(changed.sum()), changed_low=int(low.sum()),
        changed_high=int((changed & high).sum()), planes=planes, frames_audit=audits,
        scope='per-view finite patches; validation on opposite-parity views of same joint inference'))


if __name__ == '__main__':
    main()
