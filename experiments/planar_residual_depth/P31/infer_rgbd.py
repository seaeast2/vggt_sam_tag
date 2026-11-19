"""Run VGGT-Ω on a manifest of RGB frames, without reading the sensor depth."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

from run import write_json


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / 'third_party/vggt-omega'
sys.path.insert(0, str(REPO))
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera


def vggt_source_sha256():
    digest = hashlib.sha256()
    for path in sorted((REPO / 'vggt_omega').rglob('*.py')):
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    size = tuple(manifest['size_wh'])
    frames = []
    for frame in manifest['frames']:
        path = args.manifest.parent / frame['rgb']
        if hashlib.sha256(path.read_bytes()).hexdigest() != frame['rgb_sha256']:
            raise ValueError(f'RGB hash mismatch for {path}')
        with Image.open(path) as source:
            frames.append(np.asarray(source.convert('RGB').resize(size, Image.Resampling.BICUBIC)))
    rgb = np.stack(frames)
    checkpoint = ROOT / 'models/vggt_omega/vggt_omega_1b_512.pt'
    args.out.mkdir(parents=True)
    start = time.monotonic()
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True))
    model.cuda()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(rgb).permute(0, 3, 1, 2).cuda().float() / 255)
        ext, intrinsics = encoding_to_camera(prediction['pose_enc'], prediction['images'].shape[-2:])
    depth = prediction['depth'][0, ..., 0].float().cpu().numpy()
    confidence = prediction['depth_conf'][0].float().cpu().numpy().reshape(depth.shape)
    cameras = intrinsics[0].double().cpu().numpy() @ ext[0].double().cpu().numpy()
    if not (np.isfinite(depth).all() and (depth > 0).all() and np.isfinite(confidence).all()):
        raise ValueError('Invalid VGGT-Ω prediction')
    y, x = np.indices(depth.shape[1:])
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    xyz, keep = [], []
    kernel = np.ones((3, 3), 'uint8')
    for z, camera in zip(depth, cameras, strict=True):
        linear = camera[:, :3]
        center = np.linalg.solve(linear, -camera[:, 3])
        rays = pixels @ np.linalg.inv(linear).T
        xyz.append(center + rays * z[..., None])
        hi = cv2.dilate(z, kernel, borderType=cv2.BORDER_REPLICATE)
        lo = cv2.erode(z, kernel, borderType=cv2.BORDER_REPLICATE)
        keep.append((hi - lo) / np.maximum(z, 1e-6) <= .03)
    np.savez_compressed(args.out / 'input.npz', rgb=rgb, xyz=np.asarray(xyz, dtype='float32'),
                        depth=depth, confidence=confidence, cameras=cameras, keep=np.stack(keep))
    with checkpoint.open('rb') as stream:
        checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    write_json(args.out / 'inference.json', dict(
        manifest=str(args.manifest), frames=len(frames), size_wh=size,
        execution_context='host outside sandbox', torch=torch.__version__,
        cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic() - start,
        workspace_head=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                               text=True).strip(),
        vggt_source_sha256=vggt_source_sha256(),
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=checkpoint_sha))


if __name__ == '__main__':
    main()
