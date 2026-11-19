"""Measure the coverage and depth errors of the frozen edge filter."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'P31'))
from evaluate import registered_reference
from run import export_ply, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run
    summary = json.loads((run / 'summary.json').read_text())
    evaluation = json.loads((run / 'evaluation.json').read_text())
    with np.load(summary['source']) as data:
        rgb, points, raw, keep = (data[k] for k in ('rgb','xyz','depth','keep'))
    gt, _, _ = registered_reference(Path(evaluation['dataset']), rgb)
    y, x = np.indices(raw.shape[1:])
    calibration = (x // 16 + 3 * (y // 16)) % 5 == 2
    valid = (gt > 0) & ~calibration
    absolute = abs(raw * evaluation['scale'] - gt)
    outlier = absolute > .10 * gt

    def measure(region):
        pixels = int(region.sum())
        return dict(pixels=pixels, above_10pct=int((outlier & region).sum()),
                    mae_cm=float(100 * absolute[region].mean()))

    with np.load(run/'pointmaps.npz') as maps:
        changed = valid & maps['changed']
        projection_outliers = {
            name: int(((abs(maps[name]*evaluation['scale']-gt) > .10*gt) & changed).sum())
            for name in ('raw','low_only','all_confidence')
        }
    for name, marked in [('raw_unfiltered', None), ('edge_removed_orange', ~keep)]:
        path = run/f'{name}.ply'
        if not path.exists():
            export_ply(path, points, rgb, np.ones_like(keep), marked)
    write_json(run/'edge_audit.json',dict(
        scale=evaluation['scale'], all_pixels=int(raw.size), rendered_kept=int(keep.sum()),
        sensor_all=measure(valid), sensor_retained=measure(valid & keep),
        sensor_removed=measure(valid & ~keep),
        same_projection_candidates=dict(pixels=int(changed.sum()), above_10pct=projection_outliers),
        cutoff='absolute depth error > 10% of registered sensor depth'))


if __name__ == '__main__':
    main()
