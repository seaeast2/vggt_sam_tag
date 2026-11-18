"""Reinfer identical KITTI source crops with larger input rasters."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'third_party/vggt-omega'))
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera
SOURCE=ROOT/'results/planar_residual_depth/P16-real-20261118-001/outdoor-test'
out=Path(sys.argv[1])
out.mkdir(exist_ok=False,parents=True)
assert os.environ.get('P10_HOST_GPU')=='1'
views=json.loads((SOURCE/'case.json').read_text())['images']
checkpoint=ROOT/'models/vggt_omega/vggt_omega_1b_512.pt'
model=VGGTOmega().eval()
model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True))
model.cuda()
for width in [512,640]:
    folder=out/f'R{width}'
    folder.mkdir()
    start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    images=[]
    for v in views:
        path=ROOT/v['source']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==v['source_sha256']
        with Image.open(path) as im:
            w,h=im.size
            crop_w=min(w,2*h)
            crop_h=min(h,2*w)
            x,y=(w-crop_w)//2,(h-crop_h)//2
            image=im.convert('RGB').crop((x,y,x+crop_w,y+crop_h))
            images.append(np.array(image.resize((width,width//2),Image.Resampling.BICUBIC)))
    rgb=np.stack(images)
    images=torch.from_numpy(rgb).permute(0,3,1,2).cuda().float()/255
    with torch.inference_mode():
        pred=model(images)
        ext,k=encoding_to_camera(pred['pose_enc'],pred['images'].shape[-2:])
    raw=pred['depth'][0,...,0].float().cpu().numpy()
    assert np.isfinite(raw).all() and (raw>0).all()
    conf=pred['depth_conf'][0].float().cpu().numpy().reshape(raw.shape)
    pose=np.tile(np.eye(4,dtype='float32'),(len(raw),1,1))
    pose[:,:3]=ext[0].float().cpu().numpy()
    np.savez_compressed(folder/'prediction.npz',raw=raw,rgb=rgb,confidence=conf,
                        model_K=k[0].float().cpu().numpy(),model_pose=pose)
    meta=dict(width=width,height=width//2,frames=len(raw),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
              seconds=time.monotonic()-start,checkpoint=str(checkpoint.relative_to(ROOT)),device=torch.cuda.get_device_name())
    (folder/'inference.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(width,meta,flush=True)
    del pred,ext,k,images
    torch.cuda.empty_cache()
