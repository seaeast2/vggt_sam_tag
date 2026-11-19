"""Conservative frustum queries over finite patch bounds."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Node:
    lower: np.ndarray
    upper: np.ndarray
    items: np.ndarray
    children: tuple = ()
    axis: int = 0
    split: float = 0.


def frustum(camera, shape, margin=0):
    h, w = shape
    x, y, z = camera
    return np.stack((x + margin * z, (w - 1 + margin) * z - x,
                     y + margin * z, (h - 1 + margin) * z - y, z))


def intersects(lower, upper, planes):
    normal = planes[:, :3]
    extent = np.where(normal >= 0, np.asarray(upper)[..., None, :],
                      np.asarray(lower)[..., None, :])
    return np.all((extent * normal).sum(axis=-1) + planes[:, 3] >= -1e-9, axis=-1)


class PatchIndex:
    def __init__(self, patches, kind='bvh'):
        if kind not in ('bvh', 'bsp', 'linear'):
            raise ValueError(f'Unknown index {kind}')
        self.kind = kind
        self.ids = np.array([p['id'] for p in patches], dtype=int)
        self.bounds = np.asarray([p['bounds'] for p in patches], dtype=float).reshape(-1, 2, 3)
        self.root = self.build(np.arange(len(patches))) if len(patches) and kind != 'linear' else None

    def build(self, items, depth=0):
        boxes = self.bounds[items]
        lower, upper = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
        if len(items) <= 6 or depth >= 16:
            return Node(lower, upper, items)
        centers = boxes.mean(axis=1)
        axis = int(np.argmax(np.ptp(centers, axis=0)))
        split = float(np.median(centers[:, axis]))
        if self.kind == 'bvh':
            ordered = items[np.argsort(centers[:, axis], kind='stable')]
            left, right = np.array_split(ordered, 2)
        else:
            left = items[boxes[:, 0, axis] <= split]
            right = items[boxes[:, 1, axis] >= split]
            # Limit duplication in rooms with large wall, ceiling and floor patches.
            if min(len(left), len(right)) == 0 or max(len(left), len(right)) == len(items):
                return Node(lower, upper, items)
            if len(left) + len(right) > 1.45 * len(items):
                return Node(lower, upper, items)
        return Node(lower, upper, np.empty(0, dtype=int),
                    (self.build(left, depth + 1), self.build(right, depth + 1)), axis, split)

    def query(self, camera, shape, margin=0):
        planes = frustum(camera, shape, margin)
        if not len(self.ids):
            return []
        if self.kind == 'linear':
            return sorted(self.ids[intersects(self.bounds[:, 0], self.bounds[:, 1], planes)].tolist())
        center = -np.linalg.solve(camera[:, :3], camera[:, 3])
        pending, hits = [self.root], set()
        while pending:
            node = pending.pop()
            if not intersects(node.lower, node.upper, planes):
                continue
            if node.children:
                children = node.children
                if center[node.axis] <= node.split:
                    children = children[::-1]
                pending.extend(children)
            else:
                hits.update(node.items.tolist())
        if not hits:
            return []
        items = np.array(sorted(hits))
        boxes = self.bounds[items]
        return sorted(self.ids[items[intersects(boxes[:, 0], boxes[:, 1], planes)]].tolist())

    def stats(self):
        stack, nodes, leaves, entries, array_bytes = [self.root] if self.root else [], 0, 0, 0, 0
        while stack:
            node = stack.pop()
            nodes += 1
            leaves += not node.children
            entries += len(node.items)
            array_bytes += node.items.nbytes + node.lower.nbytes + node.upper.nbytes
            stack.extend(node.children)
        return dict(nodes=nodes, leaves=leaves, leaf_entries=entries,
                    numeric_bytes=array_bytes + self.bounds.nbytes + self.ids.nbytes)


def project_masks(points_by_patch, camera, shape, selected, margin=0, footprints=None):
    """Rasterize finite planar rings and z-test their ray/plane intersections."""
    h, w = shape
    zbuffer = np.full((h,w), np.inf)
    pixels = []
    linear = camera[:, :3]
    ray_matrix = np.linalg.inv(linear).T
    center = -np.linalg.solve(linear, camera[:, 3])
    for patch in selected:
        if patch not in points_by_patch:
            continue
        before = len(pixels)
        for observation in (footprints or {}).get(patch, ()):
            rings = observation['rings']
            mask = np.zeros((h,w), np.uint8)
            drawn = set()
            for i, ring in enumerate(rings):
                coords = np.asarray(ring['xyz'])
                parent = ring['parent']
                if len(coords) < 3 or (parent != -1 and parent not in drawn):
                    continue
                projected = coords @ linear.T + camera[:, 3]
                if np.any(projected[:, 2] <= 1e-8):
                    continue
                uv = projected[:, :2] / projected[:, 2:]
                uv = np.rint(np.clip(uv, -1e6, 1e6) * 16).astype('int32')
                if abs(cv2.contourArea(uv)) < 1:
                    continue
                level = 0
                while parent != -1:
                    level += 1
                    parent = rings[parent]['parent']
                cv2.fillPoly(mask, [uv], int(level % 2 == 0), shift=4)
                drawn.add(i)
            y, x = np.nonzero(mask)
            n = np.asarray(observation['plane'][:3])
            denom = np.column_stack((x, y, np.ones(len(x)))) @ (ray_matrix @ n)
            z = np.divide(observation['plane'][3] - center @ n, denom,
                          out=np.zeros_like(denom), where=abs(denom) > 1e-8)
            visible = z > 1e-8
            if visible.any():
                x, y, z = x[visible], y[visible], z[visible]
                np.minimum.at(zbuffer, (y,x), z)
                pixels.append((patch, x, y, z))
        if len(pixels) != before:
            continue
        # Samples handle footprints clipped by the camera near plane.
        points = points_by_patch[patch]
        uvz = points @ camera[:, :3].T + camera[:, 3]
        uvz = uvz[uvz[:, 2] > 1e-8]
        uv = np.rint(uvz[:, :2] / uvz[:, 2:]).astype(int)
        x, y = uv.T
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y, z = x[valid], y[valid], uvz[valid,2]
        if len(z):
            np.minimum.at(zbuffer, (y,x), z)
            pixels.append((patch, x, y, z))
    masks = {}
    kernel = np.ones((5 + 2 * margin, 5 + 2 * margin), np.uint8)
    for patch, x, y, z in pixels:
        visible = z <= 1.025 * zbuffer[y,x]
        mask = np.zeros((h,w), np.uint8)
        mask[y[visible],x[visible]] = 1
        if mask.any():
            if patch in masks:
                masks[patch] |= mask
            else:
                masks[patch] = mask
    for patch, mask in masks.items():
        masks[patch] = cv2.dilate(mask, kernel)
    return masks
