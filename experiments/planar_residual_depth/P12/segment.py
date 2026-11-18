import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

root=Path(__file__).resolve().parents[3]
out=Path(sys.argv[1])
out.mkdir()
checkpoint=root/'models/sam/sam_vit_b_01ec64.pth'
case=json.loads((root/'results/planar_residual_depth/P10-indoor-2s-input-20261118-001/indoor/case.json').read_text())
model=sam_model_registry['vit_b'](checkpoint=checkpoint).cuda().eval()
generator=SamAutomaticMaskGenerator(model,points_per_side=16,pred_iou_thresh=.85,stability_score_thresh=.9,min_mask_region_area=100)
for i,view in enumerate(case['images']):
    masks=generator.generate(np.array(Image.open(root/view['processed']).convert('RGB')))
    np.savez_compressed(out/f'{i:02}.npz', masks=np.array([m['segmentation'] for m in masks],dtype=bool),scores=np.array([m['predicted_iou'] for m in masks]))
    print(i,len(masks),flush=True)
(out/'manifest.json').write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),frames=16,points_per_side=16,execution='host CUDA'),indent=2))
