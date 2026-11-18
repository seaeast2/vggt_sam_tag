"""Estimate a vertical wall normal from local RGB lines, then anchor its position."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# RGB line seeds in the 384x288 TUM raster. The panel seam is visible in 8 and 9.
FLOOR_EDGES = {
    7: ((332, 99), (380, 102)),
    8: ((312, 72), (378, 71)),
    9: ((310, 70), (357, 69)),
    10: ((304, 52), (334, 49)),
}
LOCAL_SEAMS = {8: ((370, 13), (357, 69)), 9: ((337, 15), (328, 67))}
CHECK_SEAMS = {
    12: ((361, 12), (349, 76)),
    13: ((377, 13), (363, 66)),
    14: ((334, 12), (324, 62)),
    15: ((344, 13), (327, 70)),
}
VERTEX = np.dtype([(key, '<f4') for key in 'xyz'] +
                  [(key, 'u1') for key in ('red', 'green', 'blue')] +
                  [('confidence', '<f4')] +
                  [(key, '<u2') for key in ('frame_id', 'pixel_u', 'pixel_v')])


def locate_line(rgb, camera, ends, floor_edge):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype('float32')
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    start, end = np.array(ends, dtype=float)
    delta = end - start
    across = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
    along = start + np.linspace(0, 1, 40)[:, None] * delta
    samples = along[:, None] + across * np.linspace(-5, 5, 41)[None, :, None]
    intensities = cv2.remap(gray, samples[..., 0].astype('float32'),
                            samples[..., 1].astype('float32'), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    if floor_edge:
        index = np.argmin(np.gradient(intensities, axis=1), axis=1)
    else:
        index = np.argmin(intensities, axis=1)
    points = samples[np.arange(len(along)), index]
    valid = (index > 2) & (index < 38)
    for _ in range(4):
        if valid.sum() < 20:
            raise ValueError(f'Insufficient RGB line support near {ends}')
        center = points[valid].mean(0)
        normal = np.linalg.svd(points[valid] - center, full_matrices=False)[2][-1]
        valid = np.abs((points - center) @ normal) < 1.1
    if valid.sum() < 20:
        raise ValueError(f'Insufficient RGB line support near {ends}')
    plane = np.r_[normal, -center @ normal] @ camera
    return plane / np.linalg.norm(plane[:3]), points[valid]


def angle(first, second):
    return float(np.degrees(np.arccos(np.clip(abs(first @ second), 0., 1.))))


def consensus(vectors):
    return np.linalg.svd(vectors, full_matrices=False)[2][0]


def vertical_axis(seams):
    planes = np.stack([plane[:3] for plane, _ in seams.values()])
    return np.linalg.svd(planes, full_matrices=True)[2][-1]


def wall_normals(edges, vertical):
    planes = np.stack([plane[:3] for plane, _ in edges.values()])
    horizontal = planes - (planes @ vertical)[:, None] * vertical
    length = np.linalg.norm(horizontal, axis=1)
    if length.min() < 1e-6:
        raise ValueError('Wall-floor edge is too close to the horizon')
    return horizontal / length[:, None], length


def export(source, target, corrected, changed):
    with source.open('rb') as stream:
        header = bytearray()
        for line in stream:
            header.extend(line)
            if line == b'end_header\n':
                break
        vertices = np.fromfile(stream, VERTEX)
    indices = tuple(vertices[key] for key in ('frame_id', 'pixel_v', 'pixel_u'))
    use = changed[indices]
    if use.sum() != changed.sum():
        raise ValueError('PLY provenance does not cover every corrected pixel')
    for key, axis in zip('xyz', range(3), strict=True):
        vertices[key][use] = corrected[..., axis][indices][use]
    for key, value in zip(('red', 'green', 'blue'), (255, 145, 20), strict=True):
        vertices[key][use] = value
    with target.open('xb') as stream:
        stream.write(header)
        vertices.tofile(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('geometry', type=Path)
    parser.add_argument('previous', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    with np.load(args.geometry / 'input.npz') as data:
        rgb, xyz, confidence, keep, cameras, raw = (
            data[k] for k in ('rgb', 'xyz', 'confidence', 'keep', 'cameras', 'depth')
        )
    previous = json.loads((args.previous / 'summary.json').read_text())
    frames = previous['test_views']
    masks = np.load(args.previous / 'sam_mask.npy')
    with np.load(args.previous / 'depths.npz') as data:
        heldout = data['spatial_test']
    edges = {i: locate_line(rgb[i], cameras[i], ends, True) for i, ends in FLOOR_EDGES.items()}
    seams = {i: locate_line(rgb[i], cameras[i], ends, False) for i, ends in LOCAL_SEAMS.items()}
    check = {i: locate_line(rgb[i], cameras[i], ends, False) for i, ends in CHECK_SEAMS.items()}

    vertical = vertical_axis(seams)
    check_vertical = vertical_axis(check)
    normals, conditioning = wall_normals(edges, vertical)
    normal = consensus(normals)
    check_normal = consensus(wall_normals(edges, check_vertical)[0])
    per_view = {i: angle(n, normal) for i, n in zip(edges, normals, strict=True)}
    leave_one_out = {
        i: angle(consensus(np.delete(normals, j, axis=0)), normal)
        for j, i in enumerate(edges)
    }
    axis_difference = angle(vertical, check_vertical)
    sensitivity = angle(normal, check_normal)
    # Pilot thresholds. These do not establish expected behavior on unrelated scenes.
    limits = dict(min_horizontal_component=.1, max_vertical_disagreement_deg=7,
                  max_normal_disagreement_deg=8, max_leave_one_out_deg=5)
    approved = (float(conditioning.min()) >= limits['min_horizontal_component']
                and axis_difference <= limits['max_vertical_disagreement_deg']
                and max(max(per_view.values()), sensitivity) <= limits['max_normal_disagreement_deg']
                and max(leave_one_out.values()) <= limits['max_leave_one_out_deg'])
    regions = np.stack([cv2.erode(mask.astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool)
                        for mask in masks]) & keep
    positions = {}
    for i in frames:
        candidates = regions[i] & ~heldout
        high = candidates & (confidence[i] >= np.median(confidence[i][candidates]))
        positions[i] = float(np.median(xyz[i][high] @ normal))
    offset = float(np.median(list(positions.values())))

    corrected = xyz.copy()
    depth = raw.copy()
    selected = np.zeros_like(keep)
    y, x = np.indices(raw.shape[1:])
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
    if approved:
        for i in frames:
            linear, translation = cameras[i, :, :3], cameras[i, :, 3]
            center = -np.linalg.solve(linear, translation)
            rays = pixels @ np.linalg.inv(linear).T
            denom = rays @ normal
            proposal = np.divide(offset - center @ normal, denom, out=np.zeros_like(denom),
                                 where=np.abs(denom) > 1e-8)
            use = regions[i] & (proposal > 0) & (np.abs(proposal - raw[i]) <= .30 * raw[i])
            depth[i][use] = proposal[use]
            corrected[i][use] = center + proposal[use, None] * rays[use]
            np.testing.assert_allclose(corrected[i][use] @ normal, offset, atol=1e-6)
            selected[i] = use
    changed = np.any(corrected != xyz, axis=-1)
    if np.any(changed & ~selected):
        raise ValueError('A vertex outside the selected region changed')
    selected = changed
    depth[~selected] = raw[~selected]
    args.out.mkdir(parents=True)
    np.save(args.out / 'sam_mask.npy', masks)
    np.savez_compressed(args.out / 'depths.npz', raw=raw, selected=selected,
                        spatial_test=heldout, local_rgb_normal=depth)
    for label in ('full', 'wall'):
        export(args.previous / f'raw_{label}.ply', args.out / f'local_rgb_normal_{label}.ply',
               corrected, selected)

    h, w = raw.shape[1:]
    sheet = Image.new('RGB', (2 * w, 2 * (h + 24)), 'white')
    title = ImageDraw.Draw(sheet)
    for j, i in enumerate(edges):
        image = Image.fromarray(rgb[i])
        draw = ImageDraw.Draw(image)
        draw.line([tuple(p) for p in edges[i][1]], fill='cyan', width=2)
        if i in seams:
            draw.line([tuple(p) for p in seams[i][1]], fill='orange', width=2)
        xpos, ypos = j % 2 * w, j // 2 * (h + 24)
        title.text((xpos + 4, ypos + 4), f'{i:02} | wall-floor edge / panel seam', fill='black')
        sheet.paste(image, (xpos, ypos + 24))
    sheet.save(args.out / 'rgb_lines.jpg', quality=94)
    summary = dict(status='EXPLORATORY_APPLIED' if approved else 'ABSTAINED',
                   source=str(args.geometry), previous=str(args.previous), test_views=frames,
                   direction_views=list(edges), normal=normal.tolist(),
                   offset_by_view=positions, offsets={'local_rgb_normal': offset},
                   old_normal_difference_deg=angle(normal, np.asarray(previous['normal'])),
                   vertical_crosscheck_deg=axis_difference, normal_sensitivity_deg=sensitivity,
                   normal_difference_by_view_deg=per_view, leave_one_wall_floor_view_out_deg=leave_one_out,
                   horizontal_component_by_view=dict(zip(edges, conditioning.tolist(), strict=True)),
                   limits=limits, plane_pixels=int(selected.sum()),
                   scope='same clip, manual RGB line regions, no GT used; edge_3x3_3pct fixed; 30% change cap')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
