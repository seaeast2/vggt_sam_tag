"""Record human review and inset M01 top-cover polygon; preserve prior masks."""
import json,hashlib
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];base=ROOT/'results/planar_residual_depth'
parent=base/'review/P1-candidates-v002';out=base/'review/P1-candidates-v003';out.mkdir(parents=True,exist_ok=False)
data=json.loads((parent/'candidates.json').read_text());sheet=Image.new('RGB',(1024,1560),'white');draw=ImageDraw.Draw(sheet)
for row,c in enumerate(data):
 old=c['polygon'];c['parent_version']='P1-candidates-v002'
 if c['id']=='M01':
  c['polygon']=[[86,206],[127,192],[165,207],[101,223]]
  c['status']='REVISED_TOP_FACE_ONLY_PENDING_HUMAN_REVIEW'
  c['review_note']='User reported side face included; polygon inset to top cover from RGB only.'
 else:
  c['status']='HUMAN_REVIEW_GENERALLY_ACCURATE'
  c['review_note']='User described other candidates as generally accurate; not an exact boundary or geometric GT certification.'
 im=Image.open(base/c['source_run']/c['image']).convert('RGB');overlay=im.copy();d=ImageDraw.Draw(overlay)
 d.line(c['polygon']+[c['polygon'][0]],fill='red',width=2)
 mask=Image.new('L',im.size,0);ImageDraw.Draw(mask).polygon(c['polygon'],fill=255);mask.save(out/f'{c["id"]}_mask.png')
 c['mask_sha256']=hashlib.sha256((out/f'{c["id"]}_mask.png').read_bytes()).hexdigest()
 if c['id']=='M01':
  ts=json.loads((base/c['source_run']/'artifacts/inputs/Adirondack/tracks.json').read_text());pts=[(t['split'],t['uv0']) for t in ts]
 else:
  sample='000000' if c['id']=='K01' else 'pipes';ts=json.loads((base/c['source_run']/'artifacts/observations/tracks.json').read_text())[sample]
  pts=[(t['split'],o['uv']) for t in ts for o in t['observations'] if o['view']==0]
 def count(poly):return {s:sum(cv2.pointPolygonTest(np.array(poly,np.float32),tuple(uv),False)>=0 for sp,uv in pts if sp==s) for s in ['fit','heldout']}
 c['support_before']=count(old);c['support']=count(c['polygon']);c['support_status']='INSUFFICIENT_SUPPORT' if c['support']['fit']<12 or c['support']['heldout']<6 else 'COUNT_GATE_PASSED_OTHER_GATES_PENDING'
 overlay.save(out/f'{c["id"]}.png');sheet.paste(im,(0,row*390+30));sheet.paste(overlay,(512,row*390+30));draw.text((4,row*390+5),f'{c["id"]}: {c["status"]}',fill='black')
 if c['id']=='M01':
  crop=(65,178,185,252);a=im.copy();ImageDraw.Draw(a).line(old+[old[0]],fill='red',width=1)
  b=overlay;pair=Image.new('RGB',(960,326),'white');pair.paste(a.crop(crop).resize((480,296)),(0,30));pair.paste(b.crop(crop).resize((480,296)),(480,30))
  dd=ImageDraw.Draw(pair);dd.text((4,5),'Previous: side face included',fill='black');dd.text((484,5),'Revised: inset top cover only',fill='black');pair.save(out/'M01_before_after.png')
 assert np.array(mask).sum()>0
sheet.save(out/'candidate_review.png');(out/'candidates.json').write_text(json.dumps(data,indent=2)+'\n')
report=ROOT/'docs/experiments/planar_residual_depth/P1/P1-audit-20261117-001/mask_review_v003.md'
report.write_text('# P1 마스크 사용자 검토 및 M01 수정\n\nK01·P01·P02는 사용자가 대체로 정확하다고 확인했다. 정밀 경계 GT나 방향 GT로 해석하지 않는다. M01은 책 옆면을 포함했다는 사용자 지적에 따라 윗면 안쪽으로 좁혔다. 수정된 M01은 재검토 대기이며 원본 후보는 보존한다.\n\n수정 좌표: '+str([c for c in data if c['id']=='M01'][0]['polygon'])+'\n\n'+ '\n'.join(f"- {c['id']}: FIT/보류 {c['support_before']} → {c['support']}; {c['support_status']}" for c in data)+'\n\n수정은 RGB만 사용했다. raw·GT·solver 결과는 선택에 사용하지 않았다. 다른 시점 마스크와 평면별 관측 지원·scale·GT 격자 검증은 여전히 미완료다. 깊이 추론과 보정을 재실행하지 않았다.\n\n산출물: `results/planar_residual_depth/review/P1-candidates-v003/`.\n')
with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write('\n- P1 마스크 검토 v003: K01/P01/P02 대체로 정확, M01 윗면으로 축소 후 재검토 대기. [기록](P1/P1-audit-20261117-001/mask_review_v003.md)\n')
print(json.dumps([{k:c[k] for k in ['id','status','support_before','support']} for c in data],indent=2))
