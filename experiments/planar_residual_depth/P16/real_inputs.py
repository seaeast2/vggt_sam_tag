"""Cache independently inferred VGGT and SAM inputs for temporal pilot splits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'third_party/vggt-omega'))
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from pack_tum import boundary

p = argparse.ArgumentParser()
p.add_argument('prepared', type=Path)
p.add_argument('run', type=Path)
p.add_argument('stage', choices=['vggt','sam'])
p.add_argument('--whole-clips', action='store_true', help='Use pre-split sample entries as complete clips')
args = p.parse_args()
if os.environ.get('P10_HOST_GPU') != '1':
    p.error('Run on host GPU')
samples = json.loads((args.prepared/'samples.json').read_text())
segments = {'train':slice(0,6), 'val':slice(7,13), 'test':slice(14,20)}
args.run.mkdir(exist_ok=True, parents=True)
jobs = []
for sample in samples:
    selections = {'':slice(None)} if args.whole_clips else segments
    for split,selection in selections.items():
        clip = dict(sample)
        clip['images'] = sample['images'][selection]
        folder = args.run/(sample['sample'] if args.whole_clips else f'{sample["sample"]}-{split}')
        folder.mkdir(exist_ok=True)
        (folder/'case.json').write_text(json.dumps(clip,indent=2)+'\n')
        jobs.append((folder,clip))
if args.stage == 'vggt':
    checkpoint = ROOT/'models/vggt_omega/vggt_omega_1b_512.pt'
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True))
    model.cuda()
else:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    checkpoint = ROOT/'models/sam/sam_vit_b_01ec64.pth'
    model = sam_model_registry['vit_b'](checkpoint=checkpoint).eval().cuda()
    generator = SamAutomaticMaskGenerator(model,points_per_side=16,pred_iou_thresh=.85,
                                           stability_score_thresh=.9,min_mask_region_area=100)
with checkpoint.open('rb') as stream:
    (args.run/f'{args.stage}_checkpoint_sha256.txt').write_text(
        hashlib.file_digest(stream,'sha256').hexdigest()+'\n')
for folder,clip in jobs:
    paths = [ROOT/v['processed'] for v in clip['images']]
    result = folder/f'{args.stage}.npz'
    if result.exists():
        continue
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        if args.stage == 'vggt':
            images = load_and_preprocess_images(list(map(str,paths)),mode='max_size',image_resolution=384).cuda()
            prediction = model(images)
            ext,k = encoding_to_camera(prediction['pose_enc'],prediction['images'].shape[-2:])
            arrays = dict(raw=prediction['depth'][0,...,0].float().cpu().numpy(),
                          confidence=prediction['depth_conf'][0].float().cpu().numpy(),
                          ext=ext[0].float().cpu().numpy(),K=k[0].float().cpu().numpy())
            del images,prediction,ext,k
        else:
            maps=[]
            for path in paths:
                rgb = np.array(Image.open(path).convert('RGB'))
                masks = generator.generate(rgb)
                maps.append(boundary(np.array([m['segmentation'] for m in masks]),*rgb.shape[:2]))
            arrays=dict(sam=maps)
    temporary = result.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**arrays)
    temporary.rename(result)
    print(args.stage,folder.name,'peak_bytes',torch.cuda.max_memory_allocated(),flush=True)
