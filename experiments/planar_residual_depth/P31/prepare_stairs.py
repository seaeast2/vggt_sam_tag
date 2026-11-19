"""Prepare synchronized Bonn stair inputs from organized RGB-D point clouds."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
from PIL import Image

from run import write_json


SOURCE = 'https://www.ais.uni-bonn.de/data/stairs/stairs_bonn.tar.bz2'


def unpack_cloud(payload):
    header, body = payload.split(b'DATA binary\n', 1)
    fields = dict(line.split(maxsplit=1) for line in header.splitlines() if not line.startswith(b'#'))
    if fields[b'FIELDS'] != b'x y z rgba' or fields[b'SIZE'] != b'4 4 4 4':
        raise ValueError('Unexpected Bonn point layout')
    shape = (int(fields[b'HEIGHT']), int(fields[b'WIDTH']))
    count = int(fields[b'POINTS'])
    if count != shape[0] * shape[1] or any(body[count * 16:]):
        raise ValueError('Unexpected point count or nonzero PCD padding')
    cloud = np.frombuffer(body, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('color', '<u4')], count=count).reshape(shape)
    z = cloud['z']
    y, x = np.indices(shape)
    valid = np.isfinite(cloud['x']) & np.isfinite(cloud['y']) & np.isfinite(z) & (z > 0)
    fit = np.flatnonzero(valid)[::29]
    cols = [cloud[key].ravel()[fit] / z.ravel()[fit] for key in ('x', 'y')]
    results = [np.linalg.lstsq(np.column_stack((col, np.ones_like(col))), pixel.ravel()[fit],
                               rcond=None)[0] for col, pixel in zip(cols, (x, y), strict=True)]
    errors = [abs(a * col + b - pixel.ravel()[fit]).max()
              for (a, b), col, pixel in zip(results, cols, (x, y), strict=True)]
    if max(errors) > .01:
        raise ValueError(f'PCD raster is inconsistent with a pinhole camera: {errors}')
    color = np.stack([(cloud['color'] >> shift).astype('uint8') for shift in (16, 8, 0)], -1)
    k = [float(results[0][0]), float(results[1][0]), float(results[0][1]), float(results[1][1])]
    return z, color, valid, k, float(max(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--indices', nargs='+', type=int, required=True)
    args = parser.parse_args()
    members = {f'frame_{i:04d}.pcd' for i in args.indices}
    payloads = {}
    with tarfile.open(args.archive, 'r|bz2') as archive:
        for item in archive:
            name = PurePosixPath(item.name).name
            if item.isfile() and name in members:
                payloads[name] = archive.extractfile(item).read()
                if payloads.keys() == members:
                    break
    if payloads.keys() != members:
        raise ValueError(f'Missing members {sorted(members - payloads.keys())}')
    frames, all_k = [], []
    args.out.mkdir(parents=True)
    for i in args.indices:
        stem = f'frame_{i:04d}'
        pcd = payloads[f'{stem}.pcd']
        depth, cloud_rgb, valid, k, error = unpack_cloud(pcd)
        color_path, depth_path = args.out / f'{stem}.png', args.out / f'{stem}-depth.png'
        # Use the color embedded in the organized cloud, including RGB at missing-depth pixels.
        Image.fromarray(cloud_rgb).save(color_path)
        mm = np.rint(np.where(valid, depth, 0) * 1000)
        if mm.max() >= 65535:
            raise ValueError('Depth exceeds uint16 representation')
        Image.fromarray(mm.astype('uint16')).save(depth_path)
        frames.append(dict(index=i, rgb=color_path.name, depth=depth_path.name,
                           rgb_sha256=hashlib.sha256(color_path.read_bytes()).hexdigest(),
                           pcd_sha256=hashlib.sha256(pcd).hexdigest(),
                           depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                           pcd_max_pinhole_error_px=error))
        all_k.append(k)
    np.testing.assert_allclose(all_k, np.tile(all_k[0], (len(all_k), 1)), atol=.01)
    write_json(args.out / 'manifest.json', dict(
        dataset='Bonn Staircase', url=SOURCE, units_per_meter=1000, source_size_wh=[640, 480],
        size_wh=[384, 288], rgb_alignment='per-pixel RGB embedded in organized PCD',
        color_k_fx_fy_cx_cy=all_k[0], frames=frames))


if __name__ == '__main__':
    main()
