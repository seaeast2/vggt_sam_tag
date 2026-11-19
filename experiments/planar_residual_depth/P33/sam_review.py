"""Run independent SAM automatic segmentation on every P33 RGB frame."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'checkpoint', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    with np.load(args.source) as source:
        images = source['rgb']
    args.out.mkdir(parents=True)
    model = sam_model_registry['vit_b'](checkpoint=str(args.checkpoint)).eval().cuda()
    settings = dict(points_per_side=16, pred_iou_thresh=.85,
                    stability_score_thresh=.9, min_mask_region_area=100)
    generator = SamAutomaticMaskGenerator(model, **settings)
    colors = np.random.default_rng(33).integers(55, 255, (1024, 3), dtype='uint8')
    h, w = images.shape[1:3]
    panels, records = [], []
    for i, rgb in enumerate(images):
        regions = sorted(generator.generate(rgb), key=lambda region: -region['area'])
        masks = np.array([r['segmentation'] for r in regions], dtype=bool).reshape(-1,h,w)
        np.savez_compressed(args.out/f'frame_{i:02}.npz', masks=masks)
        overlay = rgb.copy()
        for j, mask in enumerate(masks):
            color = colors[j % len(colors)]
            interior = cv2.erode(mask.astype('uint8'), np.ones((3,3), np.uint8)).astype(bool)
            overlay[mask] = np.rint(.50 * rgb[mask] + .50 * color).astype('uint8')
            overlay[mask & ~interior] = color
        panel = Image.new('RGB', (2*w, h+26), '#111820')
        ImageDraw.Draw(panel).text((8,7),f'Frame {i:02} | RGB',fill='white')
        ImageDraw.Draw(panel).text((w+8,7),f'SAM ViT-B | {len(masks)} masks',fill='white')
        panel.paste(Image.fromarray(rgb),(0,26))
        panel.paste(Image.fromarray(overlay),(w,26))
        panel.save(args.out/f'frame_{i:02}.png')
        panels.append(panel)
        records.append(dict(frame=i,masks=len(masks),coverage=float(masks.any(axis=0).mean())))
        print(records[-1],flush=True)
    for first in range(0,len(panels),5):
        group = panels[first:first+5]
        sheet = Image.new('RGB',(2*w,(h+26)*len(group)))
        for j,panel in enumerate(group):
            sheet.paste(panel,(0,(h+26)*j))
        sheet.save(args.out/f'frames_{first:02}_{first+len(group)-1:02}.jpg',quality=94)
    with args.checkpoint.open('rb') as f:
        digest = hashlib.file_digest(f,'sha256').hexdigest()
    manifest = dict(source=str(args.source), model='SAM ViT-B', checkpoint_sha256=digest,
                    settings=settings, frames=records, colors='independent display colors per frame')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
