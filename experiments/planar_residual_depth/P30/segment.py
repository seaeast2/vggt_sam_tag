"""Segment the tracked KITTI bollard using RGB-only crop and point prompts."""
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from protocol import PROMPTS
from segment_anything import SamPredictor, sam_model_registry

p = argparse.ArgumentParser()
p.add_argument('prediction', type=Path)
p.add_argument('out', type=Path)
a = p.parse_args()
assert os.environ.get('P10_HOST_GPU') == '1'
checkpoint = Path('models/sam/sam_vit_b_01ec64.pth')
predictor = SamPredictor(sam_model_registry['vit_b'](checkpoint=str(checkpoint)).eval().cuda())
rgb = np.load(a.prediction)['rgb']
mask = np.zeros(rgb.shape[:3], bool)
# Column at the cap, column at the foot, and cap and foot rows.
for i, (top_x, bottom_x, top, bottom) in enumerate(PROMPTS):
    left, right = min(top_x,bottom_x)-24, max(top_x,bottom_x)+24
    upper, lower = top-15, bottom+15
    crop = rgb[i,upper:lower,left:right]
    zoom = 5
    predictor.set_image(cv2.resize(crop,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_CUBIC))
    y = np.linspace(top+6,bottom-6,5)
    x = np.interp(y,[top,bottom],[top_x,bottom_x])
    points = np.concatenate([np.c_[x,y],np.c_[x-8,y],np.c_[x+8,y]])
    masks, scores, _ = predictor.predict(
        point_coords=(points-[left,upper])*zoom,
        point_labels=np.repeat([1,0,0],5),
        box=(np.array([min(x)-6,top-2,max(x)+6,bottom+2])-[left,upper,left,upper])*zoom,
        multimask_output=True)
    small = cv2.resize(masks[scores.argmax()].astype('uint8'), (right-left,lower-upper),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    mask[i,upper:lower,left:right] = small
a.out.mkdir(parents=True)
np.save(a.out/'sam_mask.npy',mask)
print(mask.sum(axis=(1,2)))
