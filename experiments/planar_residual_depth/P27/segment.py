"""SAM wall masks guided by RGB-only wall panel prompts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
from segment_anything import SamPredictor,sam_model_registry

assert os.environ.get('P10_HOST_GPU')=='1'
root=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser()
parser.add_argument('out',type=Path)
parser.add_argument('--input',type=Path)
parser.add_argument('--preset',choices=['p27','p28'],default='p27')
args=parser.parse_args()
out=args.out
rgb=np.load(args.input or out/'input.npz')['rgb']
n,h,w=rgb.shape[:3]
# Panel interiors on the background right, bounded above the floor seam.
prompts={
12: ([(310,0),(383,0),(383,89),(310,84)],[(322,40),(349,42),(370,47)],[(340,116),(285,58)]),
13: ([(296,0),(383,0),(383,79),(296,64)],[(309,31),(345,37),(371,41)],[(345,108),(274,53)]),
14: ([(280,0),(383,0),(383,84),(280,66)],[(295,34),(331,37),(368,48)],[(339,111),(258,48)]),
15: ([(258,0),(383,0),(383,94),(258,70)],[(273,34),(311,42),(349,53)],[(331,116),(233,60)])}
if args.preset=='p28':
    prompts={
    7: ([(347,0),(383,0),(383,96),(340,94)],[(366,35),(378,60)],[(362,133),(323,60)]),
    8: ([(319,0),(383,0),(383,71),(307,71)],[(327,32),(351,33),(375,44)],[(345,103),(276,40)]),
    9: ([(319,0),(369,0),(364,65),(310,68)],[(329,28),(349,34),(358,45)],[(346,97),(294,45)])}
ckpt=root/'models/sam/sam_vit_b_01ec64.pth'
predictor=SamPredictor(sam_model_registry['vit_b'](checkpoint=str(ckpt)).eval().cuda())
output=np.zeros((n,h,w),dtype=bool)
sheet=Image.new('RGB',(w*2,(h+24)*((len(prompts)+1)//2)),'white')
draw=ImageDraw.Draw(sheet)
log={}
for j,(i,(polygon,positive,negative)) in enumerate(prompts.items()):
    roi=np.zeros((h,w),dtype=np.uint8)
    cv2.fillPoly(roi,[np.array(polygon,dtype=np.int32)],1)
    x,y=np.array(polygon).T
    box=np.array([x.min(),y.min(),x.max(),y.max()],dtype=float)
    predictor.set_image(rgb[i])
    masks=[]
    scores=[]
    for point in positive:
        result,s,_=predictor.predict(point_coords=np.array([point,*negative]),
            point_labels=np.array([1,0,0]),box=box,multimask_output=True)
        idx=max(range(len(s)),key=lambda k:float(s[k]) if result[k][point[1],point[0]] else -1)
        if result[idx][point[1],point[0]]:
            masks.append(result[idx]&roi.astype(bool))
            scores.append(float(s[idx]))
    if not masks:
        raise ValueError(f'No SAM mask at frame {i}')
    output[i]=np.logical_or.reduce(masks)
    assert output[i].sum()>=500
    arr=rgb[i].copy()
    arr[output[i]]=np.rint(.55*arr[output[i]]+.45*np.array([255,145,20])).astype('uint8')
    x,y=j%2*w,j//2*(h+24)
    sheet.paste(Image.fromarray(arr),(x,y+24))
    draw.text((x+5,y+5),f'{i:02} | SAM wall pixels {output[i].sum()}',fill='black')
    log[i]=dict(roi=polygon,positive=positive,negative=negative,scores=scores,pixels=int(output[i].sum()))
sheet.save(out/'sam_wall_masks.jpg',quality=94)
np.save(out/'sam_mask.npy',output)
(out/'sam.json').write_text(json.dumps(dict(frames=log,checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
  method='RGB annotation prompts and conservative upper-wall polygons intersected with SAM foreground proposals'),indent=2)+'\n')
