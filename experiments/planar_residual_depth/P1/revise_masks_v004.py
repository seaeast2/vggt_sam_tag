"""Apply user-requested inward movement of M01 upper vertex only."""
import json,hashlib,shutil
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[3];base=ROOT/'results/planar_residual_depth'
parent=base/'review/P1-candidates-v003';out=base/'review/P1-candidates-v004';out.mkdir(parents=True,exist_ok=False)
data=json.loads((parent/'candidates.json').read_text())
for c in data:
 c['parent_version']='P1-candidates-v003'
 if c['id']!='M01':
  for suffix in ['.png','_mask.png']:shutil.copy2(parent/(c['id']+suffix),out/(c['id']+suffix))
  continue
 old=c['polygon'];new=[p.copy() for p in old];new[1]=[127,196]
 c['polygon']=new;c['status']='REVISED_UPPER_VERTEX_PENDING_HUMAN_REVIEW';c['review_note']='User requested upper vertex further inward; moved (127,192) to (127,196), other vertices unchanged.'
 im=Image.open(base/c['source_run']/c['image']).convert('RGB');overlay=im.copy();ImageDraw.Draw(overlay).line(new+[new[0]],fill='red',width=2);overlay.save(out/'M01.png')
 mask=Image.new('L',im.size,0);ImageDraw.Draw(mask).polygon(new,fill=255);mask.save(out/'M01_mask.png');c['mask_sha256']=hashlib.sha256((out/'M01_mask.png').read_bytes()).hexdigest()
 ts=json.loads((base/c['source_run']/'artifacts/inputs/Adirondack/tracks.json').read_text())
 c['support_before']=c['support'].copy();c['support']={s:sum(cv2.pointPolygonTest(np.array(new,np.float32),tuple(t['uv0']),False)>=0 for t in ts if t['split']==s) for s in ['fit','heldout']}
 crop=(65,178,185,252);pair=Image.new('RGB',(960,326),'white')
 for col,poly in enumerate([old,new]):
  a=im.copy();ImageDraw.Draw(a).line(poly+[poly[0]],fill='red',width=1);pair.paste(a.crop(crop).resize((480,296)),(col*480,30))
 d=ImageDraw.Draw(pair);d.text((4,5),'Previous v003',fill='black');d.text((484,5),'Revised v004: upper vertex inward 4 px',fill='black');pair.save(out/'M01_before_after.png')
(out/'candidates.json').write_text(json.dumps(data,indent=2)+'\n')
report=ROOT/'docs/experiments/planar_residual_depth/P1/P1-audit-20261117-001/mask_review_v004.md'
report.write_text('# M01 상단 꼭짓점 추가 축소\n\n사용자 요청으로 상단 꼭짓점만 처리 영상 좌표 (127,192)→(127,196)으로 이동했다. 나머지 꼭짓점과 K01/P01/P02 마스크는 그대로 유지했다. v003은 보존했고 v004 M01은 재검토 대기다. RGB만 사용했으며 깊이 보정은 실행하지 않았다.\n\n수정 전후: `results/planar_residual_depth/review/P1-candidates-v004/M01_before_after.png`.\n')
print([{'id':c['id'],'support':c['support']} for c in data])
