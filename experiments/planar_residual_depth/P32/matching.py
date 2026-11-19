"""Feature retrieval and robust relative-pose verification for patch replay."""
from dataclasses import dataclass
from time import perf_counter_ns

import cv2
import numpy as np


@dataclass
class Features:
    uv: np.ndarray
    descriptors: np.ndarray
    signature: np.ndarray


def detect(images):
    orb = cv2.ORB_create(nfeatures=1600, fastThreshold=10, edgeThreshold=14)
    frames = []
    for rgb in images:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        uv = np.asarray([k.pt for k in keypoints], np.float32).reshape(-1, 2)
        if descriptors is None:
            descriptors = np.empty((0, 32), np.uint8)
        # A compact visual signature proposes places without pairwise keypoint matching.
        thumbnail = cv2.resize(gray, (16, 12), interpolation=cv2.INTER_AREA).astype(float)
        thumbnail -= thumbnail.mean()
        thumbnail /= max(np.linalg.norm(thumbnail), 1e-6)
        hist = np.concatenate([np.sqrt(np.bincount(descriptors[:, i] >> 4, minlength=16))
                               for i in (0, 7, 15, 23)])
        hist /= max(np.linalg.norm(hist), 1e-6)
        frames.append(Features(uv, descriptors, np.r_[thumbnail.ravel(), hist]))
    return frames


def ranked_places(features, query, candidates):
    signature = features[query].signature
    return sorted(candidates, key=lambda i: (-float(signature @ features[i].signature), i))


def load_features(path):
    with np.load(path) as data:
        offsets, uv, descriptors, signatures = (data[k] for k in
            ('offsets', 'uv', 'descriptors', 'signatures'))
    if (len(offsets) != len(signatures)+1 or offsets[0] != 0
            or offsets[-1] != len(uv) or len(uv) != len(descriptors)
            or np.any(np.diff(offsets) < 0)):
        raise ValueError('Invalid keyframe feature ranges')
    return [Features(uv[a:b], descriptors[a:b], signature)
            for a,b,signature in zip(offsets[:-1],offsets[1:],signatures,strict=True)]


def raster_indices(features, shape):
    h, w = shape
    uv = np.rint(features.uv).astype(int)
    return np.clip(uv[:, 1], 0, h-1), np.clip(uv[:, 0], 0, w-1)


def intrinsics(projection):
    k = cv2.decomposeProjectionMatrix(projection)[0]
    return k / k[2, 2]


def transform(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation).ravel()
    return matrix


def verify(source, target, depth, target_depth, k_source, k_target, source_use=None, target_use=None):
    start = perf_counter_ns()
    a = np.arange(len(source.uv)) if source_use is None else np.flatnonzero(source_use)
    b = np.arange(len(target.uv)) if target_use is None else np.flatnonzero(target_use)
    stats = dict(source_features=len(a), target_features=len(b), comparisons=0,
                 matches=0, inliers=0, accepted=False)
    pose = None
    if min(len(a), len(b)) >= 8:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        # Forward 2-NN ratio and reverse best match.
        forward = matcher.knnMatch(source.descriptors[a], target.descriptors[b], k=2)
        reverse = {m.queryIdx: m.trainIdx for m in
                   matcher.match(target.descriptors[b], source.descriptors[a])}
        pairs = [(a[m.queryIdx], b[m.trainIdx]) for m, n in forward
                 if m.distance < .80 * n.distance and reverse[m.trainIdx] == m.queryIdx]
        stats['comparisons'] = 2 * len(a) * len(b)
        stats['matches'] = len(pairs)
        if len(pairs) >= 8:
            i, j = np.asarray(pairs).T
            z = depth[tuple(axis[i] for axis in raster_indices(source, depth.shape))]
            good = np.isfinite(z) & (z > 0)
            uv = source.uv[i[good]]
            dest = np.ascontiguousarray(target.uv[j[good]])
            rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(k_source).T
            xyz = rays * z[good, None]
            if len(xyz) >= 8:
                ok, r, t, inliers = cv2.solvePnPRansac(
                    xyz, dest, k_target, None, iterationsCount=150,
                    reprojectionError=2.5, confidence=.995, flags=cv2.SOLVEPNP_EPNP)
                if ok and inliers is not None and len(inliers) >= 8:
                    keep = inliers.ravel()
                    r, t = cv2.solvePnPRefineLM(xyz[keep], dest[keep], k_target, None, r, t)
                    rotation = cv2.Rodrigues(r)[0]
                    z_proj = (xyz[keep] @ rotation.T + t.ravel())[:, 2]
                    target_ij = j[good][keep]
                    z_meas = target_depth[tuple(axis[target_ij] for axis in
                                                raster_indices(target, target_depth.shape))]
                    depth_valid = np.isfinite(z_meas) & (z_meas > 0)
                    consistent = depth_valid & (z_proj > 0) & (
                        np.abs(z_proj - z_meas) <= .12 * z_meas)
                    span = float(cv2.contourArea(cv2.convexHull(dest[keep])) / target_depth.size)
                    stats.update(inliers=len(keep), coverage=span,
                                 consistent_depth=int(consistent.sum()),
                                 inlier_fraction=float(len(keep) / len(xyz)))
                    if (span >= .008 and stats['inlier_fraction'] >= .30
                            and consistent.sum() >= max(6, .55 * depth_valid.sum())):
                        stats['accepted'] = True
                        pose = transform(rotation, t)
    stats['matching_ms'] = (perf_counter_ns() - start) * 1e-6
    return pose, stats
