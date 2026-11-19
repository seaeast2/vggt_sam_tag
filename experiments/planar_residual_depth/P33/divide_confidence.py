"""Apply a confidence-dependent camera-Z multiplier and preserve PLY colors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'P31'))
from evaluate import registered_reference

DTYPE = np.dtype([(key, '<f4') for key in 'xyz'] +
                 [(key, 'u1') for key in ('red', 'green', 'blue')] +
                 [(key, '<u2') for key in ('frame_id', 'pixel_u', 'pixel_v')] +
                 [('confidence', '<f4')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--bounded-multiply', action='store_true',
                        help='Map confidence p1–p99 to bounded depth multipliers')
    parser.add_argument('--depth-reduction-percent', nargs=2, type=float, default=(5., 15.),
                        metavar=('MIN', 'MAX'), help='Minimum and maximum depth reduction')
    args = parser.parse_args()
    minimum, maximum = args.depth_reduction_percent
    if not 0 <= minimum <= maximum < 100:
        parser.error('Depth reductions must satisfy 0 <= MIN <= MAX < 100')
    summary = json.loads((args.run / 'summary.json').read_text())
    evaluation = json.loads((args.run / 'evaluation.json').read_text())
    with np.load(summary['source']) as source:
        raw, confidence, xyz, cameras, rgb, keep = (
            source[key] for key in ('depth', 'confidence', 'xyz', 'cameras', 'rgb', 'keep'))
    if not (np.isfinite(confidence).all() and (confidence > 0).all()):
        raise ValueError('Division requires finite positive confidence')
    centers = np.array([np.linalg.solve(p[:, :3], -p[:, 3]) for p in cameras])
    bounds = None
    if args.bounded_multiply:
        lo, hi = np.percentile(confidence, [1, 99])
        if hi <= lo:
            raise ValueError('Confidence percentile range must be positive')
        bounds = [float(lo), float(hi)]
        multiplier = np.interp(confidence, bounds, [1 - maximum/100, 1 - minimum/100])
        stem, variant = 'depth_mul_conf', 'bounded_multiply'
    else:
        multiplier = 1. / confidence
        stem, variant = 'depth_div_conf', 'divided'
    depth = raw.astype(float) * multiplier
    points = centers[:, None, None] + (
        xyz - centers[:, None, None]) * multiplier[..., None]
    recovered_z = np.einsum('fi,fhwi->fhw', cameras[:, 2, :3], points) + cameras[:, 2, 3, None, None]
    np.testing.assert_allclose(recovered_z, depth, rtol=2e-6, atol=1e-7)
    args.out.mkdir(parents=True)
    counts = {}
    for label in ('raw', 'edge3pct'):
        with (args.run / f'confidence_{label}.ply').open('rb') as stream:
            header = []
            for line in stream:
                header.append(line)
                if line == b'end_header\n':
                    break
            vertices = np.fromfile(stream, DTYPE)
        indices = tuple(vertices[key] for key in ('frame_id', 'pixel_v', 'pixel_u'))
        np.testing.assert_array_equal(vertices['confidence'], confidence[indices])
        for j, key in enumerate('xyz'):
            np.testing.assert_array_equal(vertices[key], xyz[indices][:, j])
            vertices[key] = points[indices][:, j]
        path = args.out / f'{stem}_{label}.ply'
        with path.open('xb') as stream:
            stream.writelines(header)
            vertices.tofile(stream)
        with path.open('rb') as stream:
            assert stream.read(sum(map(len, header))) == b''.join(header)
            np.testing.assert_array_equal(np.fromfile(stream, DTYPE), vertices)
        counts[label] = len(vertices)
    np.savez_compressed(args.out / 'depths.npz', raw=raw, residual=depth-raw, **{variant: depth})
    gt, _, _ = registered_reference(Path(evaluation['dataset']), rgb)
    y, x = np.indices(raw.shape[1:])
    calibration = (x // 16 + 3 * (y // 16)) % 5 == 2
    valid = (gt > 0) & keep
    scale = evaluation['scale']
    rescale = float(np.median(gt[valid & calibration] / depth[valid & calibration]))
    test = valid & ~calibration
    metrics = {
        name: float((100 * abs(prediction[test] * factor - gt[test])).mean())
        for name, prediction, factor in (
            ('raw', raw, scale), (f'{variant}_fixed_scale', depth, scale),
            (f'{variant}_recalibrated', depth, rescale))
    }
    result = dict(parent=str(args.run), counts=counts,
                  confidence_p1_p99=bounds,
                  depth_reduction_percent=[minimum, maximum] if args.bounded_multiply else None,
                  depth_multiplier_quantiles=np.percentile(multiplier, [0, 1, 50, 99, 100]).tolist(),
                  test_pixels=int(test.sum()), MAE_cm=metrics, raw_scale=scale, modified_scale=rescale)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
