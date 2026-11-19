"""Controlled geometry checks for P31b roof folds and P31c graded roads."""
import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.linalg import lsqr

from run import write_json


def topology(nx, ny):
    y, x = np.indices((ny - 1, nx - 1))
    a = (y * nx + x).ravel()
    return np.concatenate([np.column_stack((a, a + 1, a + nx)),
                           np.column_stack((a + nx + 1, a + nx, a + 1))])


def interpolation(uv, nx, ny):
    position = uv * [nx - 1, ny - 1]
    cell = np.minimum(position.astype(int), [nx - 2, ny - 2])
    dx, dy = (position - cell).T
    a = cell[:, 1] * nx + cell[:, 0]
    first = dx + dy <= 1
    vertices = np.column_stack((np.where(first, a, a + nx + 1), a + 1, a + nx))
    weights = np.column_stack((np.where(first, 1 - dx - dy, dx + dy - 1),
                               np.where(first, dx, 1 - dy), np.where(first, dy, 1 - dx)))
    return coo_matrix((weights.ravel(), (np.repeat(np.arange(len(uv)), 3), vertices.ravel())),
                      shape=(len(uv), nx * ny)).tocsr()


def second_differences(nx, ny):
    grid = np.arange(nx * ny).reshape(ny, nx)
    along = np.stack((grid[:, :-2], grid[:, 1:-1], grid[:, 2:]), -1).reshape(-1, 3)
    across = np.stack((grid[:-2], grid[1:-1], grid[2:]), -1).reshape(-1, 3)
    triples = np.concatenate((along, across))
    return coo_matrix((np.tile([1., -2., 1.], len(triples)),
                       (np.repeat(np.arange(len(triples)), 3), triples.ravel())),
                      shape=(len(triples), nx * ny)).tocsr()


def grid(nx, ny):
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    return np.column_stack((x.ravel(), y.ravel()))


def case(out, name):
    rng = np.random.default_rng(31)
    uv = grid(121, 101)
    x, y = uv.T
    fit_mask = (np.arange(len(x)) % 4 == 0) & (x > 0) & (x < 1) & (y > 0) & (y < 1)
    noise = rng.normal(0, .025, len(x))
    if name == 'roof':
        nx, ny = 3, 2
        truth = 2 - 1.6 * abs(x - .5) + .12 * y
        raw = 2 - 3.2 * (x - .5) ** 2 + .12 * y + noise
        # The ridge and eaves represent independent geometry cues in this control.
        anchors = grid(nx, ny)
        anchor_z = 2 - 1.6 * abs(anchors[:, 0] - .5) + .12 * anchors[:, 1]
        anchor_z += rng.normal(0, .01, len(anchors))
        constraint = 30 * interpolation(anchors, nx, ny)
        target = 30 * anchor_z
    else:
        nx, ny = 13, 5
        truth = .45 * np.sin(2 * np.pi * x) + .85 * x + .06 * y
        raw = truth + .065 * np.sin(70 * x + 27 * y) + noise
        constraint = 12 * second_differences(nx, ny)
        target = np.zeros(constraint.shape[0])
    matrix = interpolation(uv, nx, ny)
    design = np.column_stack((np.ones(len(x)), uv))
    system = vstack((matrix[fit_mask], csr_matrix(constraint)))
    sol = lsqr(system, np.r_[raw[fit_mask], target], atol=1e-11, btol=1e-11)
    if sol[1] not in (1, 2):
        raise ValueError(f'Mesh fit failed with LSQR status {sol[1]}')
    heights = sol[0]
    single = design @ np.linalg.lstsq(design[fit_mask], raw[fit_mask], rcond=None)[0]
    mesh = matrix @ heights
    faces = topology(nx, ny)
    vertices = np.column_stack((grid(nx, ny), heights))
    assert np.all(np.bincount(faces.ravel(), minlength=len(vertices)) > 0)
    predicted = dict(raw=raw, single_plane=single, shared_mesh=mesh)
    mae = {key: float(np.mean(abs(value[~fit_mask] - truth[~fit_mask])))
           for key, value in predicted.items()}
    test_slope = (x >= .05) & (x <= .95) & np.isclose(y, .5)
    gradients = {key: np.diff(value[test_slope]) / np.diff(x[test_slope])
                 for key, value in {**predicted, 'reference': truth}.items()}
    slope = {key: float(np.mean(abs(value - gradients['reference'])))
             for key, value in gradients.items() if key != 'reference'}
    np.savez_compressed(out / f'{name}.npz', faces=faces, vertices=vertices, uv=uv,
                        fit=fit_mask, reference=truth, **predicted)
    return dict(vertices=len(vertices), triangles=len(faces), fit_pixels=int(fit_mask.sum()),
                heldout_pixels=int((~fit_mask).sum()), height_mae=mae, centerline_slope_mae=slope,
                constraint='synthetic ridge/eave anchors with 0.01 noise' if name == 'roof'
                else 'longitudinal/transverse second differences on shared heights')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    write_json(args.out / 'summary.json', dict(status='SYNTHETIC_CONTROL', seed=31,
                    roof=case(args.out, 'roof'), road=case(args.out, 'road')))


if __name__ == '__main__':
    main()
