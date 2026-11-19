"""Extract sampled, synchronized OpenLORIS corridor RGB and registered depth pairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from run import write_json


SOURCE = 'https://ssd.mathworks.com/supportfiles/shared_nav_vision/data/OpenLORIS-Scene_corridor1-4.zip'
PREFIX = 'OpenLORIS-Scene_corridor1-4'
SIZE = [384, 224]
# Published 848x480 color-camera parameters for this archive.
COLOR_K = [611.45098876953125, 611.4857177734375, 433.2039794921875, 249.4730224609375]


def extract(zip_file, out, read=None, interval=None, bridge_after=(), bridge_substeps=4):
    if read is None:
        read = zip_file.read
    names = zip_file.namelist()

    def entries(folder):
        return sorted((Decimal(PurePosixPath(name).stem), name) for name in names
                      if name.startswith(f'{PREFIX}/{folder}/') and name.endswith('.png'))

    colors, depths = entries('color'), entries('aligned_depth')
    if interval is None:
        starts = {'early': len(colors) // 5, 'late': 7 * len(colors) // 10}
        windows = {label: range(start, start + 8 * 18, 18) for label, start in starts.items()}
    else:
        if interval <= 0:
            raise ValueError('Sampling interval must be positive')
        start, end = colors[0][0], colors[-1][0]
        targets = [start + interval * i for i in range(int((end - start) // interval) + 1)]
        if bridge_after:
            if min(bridge_after) < 0 or max(bridge_after) >= len(targets) - 1:
                raise ValueError('Bridge frame index is outside the sampled keyframe ranges')
            if bridge_substeps < 2:
                raise ValueError('A tracking bridge requires at least two substeps')
            targets = [targets[i] + interval * Decimal(j) / bridge_substeps
                       for i in sorted(set(bridge_after)) for j in range(1, bridge_substeps)]
        indices = [min(range(len(colors)), key=lambda j: abs(colors[j][0] - t)) for t in targets]
        windows = {'bridge' if bridge_after else 'full': list(dict.fromkeys(indices))}
    for label, indices in windows.items():
        directory = out / label
        directory.mkdir(parents=True)
        pending = []
        for index in indices:
            rgb_time, rgb = colors[index]
            depth_time, depth = min(depths, key=lambda pair: abs(pair[0] - rgb_time))
            delta = abs(depth_time - rgb_time)
            if delta > Decimal('.025'):
                raise ValueError(f'No synchronized depth for {rgb}')
            pending.append((index, rgb, depth, delta))

        def save(pair):
            index, rgb, depth, delta = pair
            paths = {}
            for role, name in [('rgb', rgb), ('depth', depth)]:
                relative = f'{role}-{PurePosixPath(name).name}'
                data = read(name)
                (directory / relative).write_bytes(data)
                paths.update({role: relative, f'{role}_archive_path': name,
                              f'{role}_sha256': hashlib.sha256(data).hexdigest()})
            return dict(index=index, delta_s=float(delta), **paths)

        with ThreadPoolExecutor(max_workers=6) as executor:
            frames = list(executor.map(save, pending))
        write_json(directory / 'manifest.json', dict(
            dataset=PREFIX, url=SOURCE, window=label, size_wh=SIZE, source_size_wh=[848, 480],
            interval_s=float(interval) if interval is not None else None,
            color_k_fx_fy_cx_cy=COLOR_K, units_per_meter=1000, frames=frames))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--interval', type=Decimal)
    parser.add_argument('--bridge-after', type=int, nargs='+', default=())
    parser.add_argument('--bridge-substeps', type=int, default=4)
    args = parser.parse_args()
    if args.bridge_after and args.interval is None:
        parser.error('--bridge-after requires --interval')
    with ZipFile(args.archive) as archive:
        extract(archive, args.out, interval=args.interval, bridge_after=args.bridge_after,
                bridge_substeps=args.bridge_substeps)


if __name__ == '__main__':
    main()
