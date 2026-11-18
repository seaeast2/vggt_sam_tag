"""Sample the existing indoor TUM clip at configurable intervals."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--interval', type=float, default=1)
parser.add_argument('--offset', type=float, default=2)
parser.add_argument('--run', default='P10-indoor-wide-input-20261118-001')
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
source = root / 'data/planar_residual_depth/P10/source/rgbd_dataset_freiburg3_structure_texture_far'
out = root / 'results/planar_residual_depth' / args.run
rgb_dir = out / 'indoor/RGB'
rows = [line.split() for line in (source / 'rgb.txt').read_text().splitlines()
        if line.strip() and not line.startswith('#')]
start = float(rows[0][0]) + args.offset
assert args.interval > 0 and args.offset >= 0
assert start + 15 * args.interval <= float(rows[-1][0])
selected = [min(rows, key=lambda row: abs(float(row[0]) - start - i * args.interval)) for i in range(16)]
assert len({stamp for stamp, _ in selected}) == 16
rgb_dir.mkdir(parents=True)
views = []
for i, (stamp, path) in enumerate(selected):
    destination = rgb_dir / f'{i:03}.png'
    image = Image.open(source / path).convert('RGB')
    assert image.size == (640, 480)
    image.resize((384, 288), Image.Resampling.BICUBIC).save(destination)
    views.append(dict(timestamp=stamp, t_sec=float(stamp)-float(selected[0][0]),
                      source=str((source/path).relative_to(root)),
                      source_sha256=hashlib.sha256((source/path).read_bytes()).hexdigest(),
                      processed=str(destination.relative_to(root)), size=[384,288]))
case = dict(sample='indoor', images=views)
(out/'indoor/case.json').write_text(json.dumps(case, indent=2))
(out/'samples.json').write_text(json.dumps([case], indent=2))
(out/'config_resolved.json').write_text(json.dumps(dict(model=dict(long_side=384), target_interval_sec=args.interval, start_offset_sec=args.offset)))
