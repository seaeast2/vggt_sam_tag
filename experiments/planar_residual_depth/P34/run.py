"""Compare masking the lowest and highest VGGT-Ω confidence pixels in RGB."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS / 'P31'))
from infer_rgbd import ROOT, VGGTOmega, encoding_to_camera, vggt_source_sha256
from evaluate import registered_reference
from run import export_ply, write_json

ARTIFACTS = ROOT / 'results/planar_residual_depth'
INDOOR = ARTIFACTS / 'P33-openloris-1s-input-20261119-001'
SOURCES = {
    'openloris': INDOOR / 'input.npz',
    'kitti': ARTIFACTS / 'P25-kitti-edge-options-20261118-001/R512/prediction.npz',
}
FRACTION = .20


def predict(model, rgb):
    with torch.inference_mode():
        image = torch.from_numpy(rgb).permute(0, 3, 1, 2).cuda().float() / 255
        output = model(image)
        pose, intrinsics = encoding_to_camera(output['pose_enc'], image.shape[-2:])
        depth = output['depth'][0, ..., 0].float().cpu().numpy()
        confidence = output['depth_conf'][0].float().cpu().numpy().reshape(depth.shape)
        camera = intrinsics[0].double().cpu().numpy() @ pose[0].double().cpu().numpy()
    if not (np.isfinite(depth).all() and (depth > 0).all() and np.isfinite(confidence).all()
            and np.isfinite(camera).all()):
        raise ValueError('Invalid model output')
    return depth, confidence, camera


def confidence_masks(confidence):
    flattened = confidence.reshape(len(confidence), -1)
    order = flattened.argsort(axis=1, kind='stable')
    count = round(flattened.shape[1] * FRACTION)
    masks = {}
    for name, indices in [('hide_low', order[:, :count]), ('hide_high', order[:, -count:])]:
        mask = np.zeros_like(flattened, dtype=bool)
        np.put_along_axis(mask, indices, True, axis=1)
        masks[name] = mask.reshape(confidence.shape)
    return masks


def xyz_from_depth(depth, cameras):
    y, x = np.indices(depth.shape[1:])
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    xyz = []
    for z, camera in zip(depth, cameras, strict=True):
        linear = camera[:, :3]
        center = np.linalg.solve(linear, -camera[:, 3])
        xyz.append(center + (pixels @ np.linalg.inv(linear).T) * z[..., None])
    return np.asarray(xyz, dtype='float32')


def review(folder, inputs, depths, masks, gt):
    names = list(inputs)
    height, width = depths['raw'].shape[1:]
    lo, hi = np.quantile(depths['raw'], [.02, .98])
    rows = folder / 'frames'
    rows.mkdir()

    def heatmap(depth, hidden):
        intensity = np.clip((depth - lo) / (hi - lo), 0, 1)
        heat = cv2.applyColorMap(np.uint8(255 * (1 - intensity)), cv2.COLORMAP_TURBO)[..., ::-1].copy()
        heat[hidden] = (24, 24, 24)
        return Image.fromarray(heat)

    frames = []
    for i in range(len(depths['raw'])):
        sheet = Image.new('RGB', (3 * width, 2 * height + 54), (22, 27, 33))
        draw = ImageDraw.Draw(sheet)
        for j, name in enumerate(names):
            x = j * width
            draw.text((x + 8, 7), f'{i:02d}  {name} / RGB input', fill='white')
            sheet.paste(Image.fromarray(inputs[name][i]), (x, 25))
            draw.text((x + 8, height + 32), 'Depth / warm = near, dark = hidden', fill='white')
            sheet.paste(heatmap(depths[name][i], masks[name][i]), (x, height + 54))
        sheet.save(rows / f'{i:02d}.jpg', quality=94)
        frames.append(sheet)
    indices = sorted({0, len(frames) // 2, len(frames) - 1})
    overview = Image.new('RGB', (frames[0].width, len(indices) * frames[0].height))
    for row, i in enumerate(indices):
        overview.paste(frames[i], (0, row * frames[0].height))
    overview.save(folder / 'comparison.jpg', quality=93)


def metric(depth, gt, use):
    error = np.abs(depth[use] - gt[use])
    return dict(pixels=int(use.sum()), mae_cm=float(100 * error.mean()),
                above_10pct=float(np.mean(error > .10 * gt[use])))


def run_scene(model, scene, source, folder):
    folder.mkdir()
    with np.load(source) as data:
        rgb = data['rgb']
        cached = data['depth' if scene == 'openloris' else 'raw']
    predictions = {'raw': predict(model, rgb)}
    raw, confidence, cameras = predictions['raw']
    masks = {'raw': np.zeros_like(raw, dtype=bool), **confidence_masks(confidence)}
    inputs = {'raw': rgb}
    for name in ('hide_low', 'hide_high'):
        masked = rgb.copy()
        masked[masks[name]] = 0
        inputs[name] = masked
        predictions[name] = predict(model, masked)
        print(f'{scene}: {name} done', flush=True)

    common = ~(masks['hide_low'] | masks['hide_high'])
    y, x = np.indices(raw.shape[1:])
    scale_fit = (x // 16 + 3 * (y // 16)) % 5 == 2
    comparison = common & ~scale_fit
    gt = None
    if scene == 'openloris':
        manifest = json.loads((INDOOR / 'inference.json').read_text())['manifest']
        gt, _, _ = registered_reference(ROOT / manifest, rgb)
    summary = dict(source=str(source.relative_to(ROOT)), frames=len(raw), raster_wh=list(raw.shape[:0:-1]),
                   baseline_cache_median_relative_delta=float(np.median(abs(raw / cached - 1))), variants={})
    if gt is not None:
        gt_scale = float(np.median((gt / raw)[(gt > 0) & scale_fit & common]))
        summary['raw_meters_scale'] = gt_scale

    depths = {}
    archive = {'rgb': rgb, 'raw_confidence': confidence}
    for name, (depth, conf, camera) in predictions.items():
        hidden = masks[name]
        visible = ~hidden
        scale = float(np.median((raw / depth)[common & scale_fit]))
        aligned = depth * scale
        depths[name] = aligned
        change = np.abs(aligned / raw - 1)
        xyz = xyz_from_depth(aligned, cameras)
        export_ply(folder / f'{name}.ply', xyz, rgb, visible)
        if name != 'raw':
            export_ply(folder / f'{name}_changes_orange.ply', xyz, rgb, visible, change > .10)
            export_ply(folder / f'raw_matching_{name}_mask.ply',
                       xyz_from_depth(raw, cameras), rgb, visible)
        row = dict(scale_to_raw=scale, masked_pixels=int(hidden.sum()), vertices=int(visible.sum()),
                   common_median_change_pct=float(100 * np.median(change[comparison])),
                   visible_changed_above_10pct=int(np.count_nonzero(visible & (change > .10))))
        if gt is not None:
            valid = (gt > 0) & ~scale_fit
            row['gt_common'] = metric(aligned * gt_scale, gt, valid & common)
            row['gt_visible'] = metric(aligned * gt_scale, gt, valid & visible)
            row['gt_raw_on_same_visible'] = metric(raw * gt_scale, gt, valid & visible)
        summary['variants'][name] = row
        archive.update({f'{name}_depth_native': depth, f'{name}_depth_aligned': aligned,
                        f'{name}_mask': hidden, f'{name}_confidence': conf,
                        f'{name}_cameras_native': camera})
    if gt is not None:
        archive['gt_depth_m'] = gt.astype('float32')
    np.savez_compressed(folder / 'predictions.npz', **archive)
    review(folder, inputs, depths, masks, gt)
    write_json(folder / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    checkpoint = ROOT / 'models/vggt_omega/vggt_omega_1b_512.pt'
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True))
    model.cuda()
    for scene, source in SOURCES.items():
        run_scene(model, scene, source, args.out / scene)
    with checkpoint.open('rb') as stream:
        checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    write_json(args.out / 'manifest.json', dict(model='VGGT-Omega', device=torch.cuda.get_device_name(),
               cuda=torch.version.cuda, torch=torch.__version__, checkpoint_sha256=checkpoint_sha,
               source_sha256=vggt_source_sha256(), execution_context='host outside sandbox',
               masked_fraction_per_frame=FRACTION, fill_rgb=[0, 0, 0],
               ply_cameras='fixed to per-scene raw prediction', orange='relative depth change > 10%'))


if __name__ == '__main__':
    main()
