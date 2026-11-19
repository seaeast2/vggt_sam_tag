"""Evaluate P31a on timestamp-matched TUM depth using a common scale and mask."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from run import fit, write_json


def rows(dataset, filename):
    return [line.split() for line in (dataset / filename).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def reference(dataset, rgb):
    colors, depths = rows(dataset, 'rgb.txt'), rows(dataset, 'depth.txt')
    h, w = rgb.shape[1:3]
    images, matched = [], []
    for i, image in enumerate(rgb):
        target = float(colors[0][0]) + 2 * i
        timestamp, color = min(colors, key=lambda row: abs(float(row[0]) - target))
        sensor_time, sensor = min(depths, key=lambda row: abs(float(row[0]) - float(timestamp)))
        with Image.open(dataset / color) as source:
            np.testing.assert_array_equal(
                image, np.asarray(source.convert('RGB').resize((w, h), Image.Resampling.BICUBIC)))
        if abs(float(sensor_time) - float(timestamp)) > .03:
            raise ValueError(f'Missing synchronized reference for frame {i}')
        with Image.open(dataset / sensor) as source:
            images.append(np.asarray(source.resize((w, h), Image.Resampling.NEAREST)) / 5000)
        matched.append(dict(frame=i, rgb=color, sensor=sensor,
                            delta_s=abs(float(sensor_time) - float(timestamp))))
    return np.stack(images), matched


def metrics(errors, mask):
    result = dict(pixels=int(mask.sum()))
    if mask.any():
        for name, error in errors.items():
            e = error[mask]
            result[name] = dict(mae_cm=float(100 * e.mean()),
                                rmse_cm=float(100 * np.sqrt(np.mean(e ** 2))),
                                p90_cm=float(100 * np.quantile(e, .9)))
    return result


def registered_reference(manifest_path, rgb):
    manifest = json.loads(manifest_path.read_text())
    w, h = manifest['size_wh']
    images = []
    for frame, image in zip(manifest['frames'], rgb, strict=True):
        for role in ('rgb', 'depth'):
            path = manifest_path.parent / frame[role]
            if hashlib.sha256(path.read_bytes()).hexdigest() != frame[f'{role}_sha256']:
                raise ValueError(f'{role} hash mismatch for {path}')
            with Image.open(path) as source:
                if list(source.size) != manifest['source_size_wh']:
                    raise ValueError(f'Unexpected source size for {path}')
                if role == 'rgb':
                    np.testing.assert_array_equal(image, np.asarray(
                        source.convert('RGB').resize((w, h), Image.Resampling.BICUBIC)))
                else:
                    images.append(np.asarray(source.resize((w, h), Image.Resampling.NEAREST))
                                  / manifest['units_per_meter'])
    return np.stack(images), manifest['frames'], manifest


def compare_planes(planes, labels, valid, gt, cameras, xyz, corrected, scale, low_only,
                   sensor_rays=None):
    result = []
    y, x = np.indices(gt.shape[1:])
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    rotations = [cv2.decomposeProjectionMatrix(camera)[1] for camera in cameras]
    for plane in planes:
        i = plane['frame']
        region = labels[i] == plane['id']
        use = valid[i] & region
        n, offset = np.array(plane['normal']), plane['offset']
        before = abs(xyz[i, region] @ n - offset)
        after = abs(corrected[i, region] @ n - offset)
        center = np.linalg.solve(cameras[i, :, :3], -cameras[i, :, 3])
        world_rays = pixels @ np.linalg.inv(cameras[i, :, :3]).T
        intermediate = center + world_rays[region] * low_only[i, region, None]
        low_residual = abs(intermediate @ n - offset)
        stats = dict(id=plane['id'], frame=i, pixels=int(use.sum()),
                     median_residual_cm=[float(100 * scale * np.median(v))
                                         for v in (before, low_residual, after)])
        if use.sum() >= 100:
            rays = (world_rays[use] @ rotations[i].T if sensor_rays is None
                    else sensor_rays[use])
            ref = fit(rays * gt[i, use, None])
            if ref is not None:
                reference_normal, reference_offset = ref
                plane_normal = rotations[i] @ n
                ref_inliers = abs(rays @ reference_normal * gt[i, use] - reference_offset)
                stats['sensor_reference_median_residual_cm'] = float(100 * np.median(ref_inliers))
                if np.median(ref_inliers) <= .015:
                    angle_cos = np.clip(plane_normal @ reference_normal, -1, 1)
                    projected_offset = scale * (offset - center @ n)
                    stats['normal_error_deg'] = float(np.degrees(np.arccos(abs(angle_cos))))
                    stats['offset_error_cm'] = float(100 * abs(
                        projected_offset - np.copysign(1, angle_cos) * reference_offset))
        result.append(stats)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--registered-manifest', action='store_true',
                        help='Interpret dataset argument as a registered RGB-D manifest')
    parser.add_argument('--roi', type=Path, help='Optional RGB-defined polygon annotations')
    args = parser.parse_args()
    summary = json.loads((args.run / 'summary.json').read_text())
    with np.load(summary['source']) as data:
        rgb, cameras, xyz = (data[k] for k in ('rgb', 'cameras', 'xyz'))
    with np.load(args.run / 'pointmaps.npz') as data:
        raw, low_only, all_conf, keep, high, changed, spatial, labels, corrected = (
            data[k] for k in ('raw', 'low_only', 'all_confidence', 'keep', 'high',
                              'changed', 'spatial_holdout', 'labels', 'xyz'))
    sensor_rays = None
    if args.registered_manifest:
        gt, matched, manifest = registered_reference(args.dataset, rgb)
        fx, fy, cx, cy = manifest['color_k_fx_fy_cx_cy']
        src_w, src_h = manifest['source_size_wh']
        w, h = manifest['size_wh']
        v, u = np.indices((h, w))
        sensor_rays = np.stack((((u + .5) * src_w / w - .5 - cx) / fx,
                                ((v + .5) * src_h / h - .5 - cy) / fy,
                                np.ones_like(u)), axis=-1)
    else:
        gt, matched = reference(args.dataset, rgb)
    # A spatially separate fifth of the raster is reserved for scale.
    y, x = np.indices(raw.shape[1:])
    calibration = (x // 16 + 3 * (y // 16)) % 5 == 2
    scale_mask = (gt > 0) & keep & calibration
    scale = float(np.median(gt[scale_mask] / raw[scale_mask]))
    valid = (gt > 0) & keep & ~calibration
    errors = {name: abs(depth * scale - gt) for name, depth in
              zip(('baseline', 'low_only', 'all_confidence'), (raw, low_only, all_conf), strict=True)}
    gates = dict(visible=valid, changed=valid & changed,
                 changed_low=valid & changed & ~high, changed_high=valid & changed & high,
                 spatial_holdout=valid & spatial & changed,
                 spatial_holdout_high=valid & spatial & changed & high)
    roi_meta = None
    if args.roi:
        if not args.registered_manifest:
            parser.error('--roi requires --registered-manifest')
        annotation = json.loads(args.roi.read_text())
        roi = np.zeros(raw.shape, dtype='uint8')
        scale_xy = np.array([raw.shape[2], raw.shape[1]]) / annotation['size_wh']
        for mask, frame in zip(roi, matched, strict=True):
            polygon = annotation['polygons_by_frame_index'][str(frame['index'])]
            cv2.fillPoly(mask, [np.rint(np.asarray(polygon) * scale_xy).astype('int32')], 1)
        region = roi.astype(bool)
        name = annotation['name']
        gates[name] = valid & region
        gates[f'{name}_changed'] = valid & region & changed
        roi_meta = dict(path=str(args.roi), pixels=int(region.sum()),
                        visible_pixels=int((keep & region).sum()),
                        changed_pixels=int((region & changed).sum()))
    frame_stats = [{key: metrics({name: val[i] for name, val in errors.items()}, mask[i])
                    for key, mask in gates.items()} for i in range(len(raw))]
    result = dict(scale=scale, scale_pixels=int(scale_mask.sum()), dataset=str(args.dataset),
                  matches=matched, roi=roi_meta,
                  pooled={key: metrics(errors, mask) for key, mask in gates.items()},
                  frames=frame_stats, planes=compare_planes(
                      summary['planes'], labels, valid, gt, cameras, xyz, corrected, scale,
                      low_only, sensor_rays))
    write_json(args.run / 'evaluation.json', result)
    print(json.dumps(result['pooled'], indent=2))


if __name__ == '__main__':
    main()
