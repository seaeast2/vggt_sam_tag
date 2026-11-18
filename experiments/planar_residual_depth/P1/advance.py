"""Freeze human review; separate high-resolution RGB-only support diagnostic."""
import json,hashlib,shutil,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from observations import tracks
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 seq=1;prefix='approved-planes-P1a-'+time.strftime('%Y%m%d')+'-'
 while (BASE/(prefix+f'{seq:03}')).exists():seq+=1
 rid=prefix+f'{seq:03}';out=BASE/rid;out.mkdir(parents=True,exist_ok=False)
 (out/'source').mkdir()
 for name in ['advance.py','observations.py']:shutil.copy2(Path(__file__).parent/name,out/'source'/name)
 cs=json.loads((BASE/'review/P1-candidates-v004/candidates.json').read_text())
 for c in cs:
  if c['id']=='M01':c['status']='HUMAN_APPROVED_TOP_FACE';c['review_note']='User confirmed v004 polygon fully inside planar top cover.'
  c['approved_mask_sha256']=sha(BASE/'review/P1-candidates-v004'/f'{c["id"]}_mask.png')
 dump(out/'artifacts/approved_candidates.json',cs)
 dump(out/'config_resolved.json',{'matching_long_side_cap':1024,'never_upsample':True,'ratio':.75,'epipolar_px_in_512_grid':1,
 'normal_FIT_min':12,'normal_holdout_min':6,'area_fraction_min':.1,'median_parallax_deg_min':1,
 'scope':'P1a observational support only; no GT or raw depth used; baseline P1 support verdicts unchanged'})
 datasets={};summaries=[]
 for c in cs:
  sample='Adirondack' if c['id']=='M01' else '000000' if c['id']=='K01' else 'pipes'
  key=(c['source_run'],sample)
  if key in datasets:continue
  source=BASE/c['source_run'];entry=next(s for s in json.loads((source/'samples.json').read_text()) if s['sample']==sample)
  images=[]
  for idx,old in enumerate(entry['images']):
   image=Image.open(ROOT/old['source']).convert('RGB');w,h=image.size;cw=min(w,round(h*2));ch=min(h,round(w*2));left=(w-cw)//2;top=(h-ch)//2
   factor=min(1,1024/max(cw,ch));nw=round(cw*factor);nh=round(ch*factor)
   image=image.crop((left,top,left+cw,top+ch)).resize((nw,nh),Image.Resampling.BICUBIC)
   dest=out/'artifacts/matching_rgb'/sample/f'view{idx}.png';dest.parent.mkdir(parents=True,exist_ok=True);image.save(dest)
   An=np.array([[nw/cw,0,(-left+.5)*nw/cw-.5],[0,nh/ch,(-top+.5)*nh/ch-.5],[0,0,1.]])
   item=dict(old);item['processed']=str(dest.relative_to(ROOT));item['keypoint_to_calibrated']=(np.array(old['A'])@np.linalg.inv(An)).tolist();images.append(item)
   test=np.array([[0.,0.,1],[100,150,1.]]).T;A=np.array(item['keypoint_to_calibrated']);assert np.max(abs(np.linalg.inv(A)@(A@test)-test))<1e-9
  ts=tracks(images);datasets[key]=(entry,ts);dump(out/'artifacts/tracks'/f'{sample}.json',ts)
 for c in cs:
  sample='Adirondack' if c['id']=='M01' else '000000' if c['id']=='K01' else 'pipes';entry,ts=datasets[(c['source_run'],sample)]
  poly=np.array(c['polygon'],np.float32);inside=[]
  for t in ts:
   obs={o['view']:np.array(o['uv']) for o in t['observations']}
   if 0 in obs and cv2.pointPolygonTest(poly,tuple(obs[0]),False)>=0:inside.append(t)
  counts={s:sum(t['split']==s for t in inside) for s in ['fit','heldout']}
  fituv=np.array([next(o['uv'] for o in t['observations'] if o['view']==0) for t in inside if t['split']=='fit'],np.float32)
  area=float(cv2.contourArea(cv2.convexHull(fituv))/cv2.contourArea(poly)) if len(fituv)>=3 else 0.
  views=entry['images'];angles=[];reprojections=[]
  for t in inside:
   if t['split']!='fit':continue
   obs=t['observations'];a=next(o for o in obs if o['view']==0);b=next(o for o in obs if o['view']!=0)
   i,j=0,b['view'];Ki,Kj=[np.array(views[k]['K']) for k in [i,j]];Ri,Rj=[np.array(views[k]['R']) for k in [i,j]];Ci,Cj=[np.array(views[k]['C']) for k in [i,j]]
   x=np.linalg.inv(Ki)@np.r_[a['uv'],1];y=np.linalg.inv(Kj)@np.r_[b['uv'],1]
   ra=Ri.T@x;rb=Rj.T@y;angles.append(float(np.degrees(np.arccos(np.clip(ra@rb/np.linalg.norm(ra)/np.linalg.norm(rb),-1,1)))))
  median=float(np.median(angles)) if angles else None
  gate=counts['fit']>=12 and counts['heldout']>=6 and area>=.1 and median is not None and median>=1
  row={'candidate':c['id'],'baseline_counts':c['support'],'P1a_counts':counts,'FIT_hull_area_fraction':area,'median_FIT_parallax_deg':median,
  'verdict':'OBSERVATION_GATES_PASSED_TARGET_MASK_PENDING' if gate else 'INSUFFICIENT_SUPPORT','cross_view_polygons':[]}
  for view in range(1,len(views)):
   matches=[]
   for t in inside:
    if t['split']!='fit':continue
    obs={o['view']:o['uv'] for o in t['observations']}
    if view in obs:matches.append((obs[0],obs[view]))
   if len(matches)<4:continue
   a=np.float32([m[0] for m in matches]);b=np.float32([m[1] for m in matches]);H,inliers=cv2.findHomography(a,b,cv2.RANSAC,2.)
   if H is None:continue
   projected=cv2.perspectiveTransform(poly[None],H)[0]
   row['cross_view_polygons'].append({'view':view,'polygon':projected.tolist(),'FIT_matches':len(matches),'homography_inliers':int(inliers.sum()),'status':'PENDING_HUMAN_REVIEW_NOT_GT'})
   img=Image.open(ROOT/views[view]['processed']).convert('RGB');d=ImageDraw.Draw(img);pp=[tuple(p) for p in projected.tolist()]
   d.line(pp+[pp[0]],fill='red',width=2);img.save(out/'artifacts'/f'{c["id"]}_view{view}_proposed.png')
  # Display genuine matched observations in both splits; green FIT, orange heldout.
  img=Image.open(ROOT/views[0]['processed']).convert('RGB');d=ImageDraw.Draw(img)
  d.line([tuple(x) for x in c['polygon']]+[tuple(c['polygon'][0])],fill='red',width=2)
  for t in inside:
   uv=next(o['uv'] for o in t['observations'] if o['view']==0);x,y=uv;color='lime' if t['split']=='fit' else 'orange';d.ellipse((x-2,y-2,x+2,y+2),outline=color,width=1)
  img.save(out/'artifacts'/f'{c["id"]}_source_tracks.png')
  dump(out/'artifacts'/f'{c["id"]}_plane_tracks.json',inside);summaries.append(row)
 dump(out/'evaluation/support.json',summaries)
 dump(out/'manifest.json',{'run_id':rid,'stage':'P1a','status':'OBSERVATION_DIAGNOSTIC_COMPLETED','parent':'P1-candidates-v004','GT_read':False,'correction_executed':False,'source_sha256':sha(Path(__file__))})
 report=ROOT/'docs/experiments/planar_residual_depth/P1a'/rid/'report.md';report.parent.mkdir(parents=True,exist_ok=True)
 lines=['# 승인 마스크 동결·원본 RGB 해상도 매칭 진단','', 'M01 v004는 사용자 승인 상태로 동결했다. K01/P01/P02의 기존 검토 기록을 유지한다. 원래 P1 지원 부족 판정은 변경하지 않았다. 별도 P1a에서 매칭 입력만 원본 RGB 긴 변 최대1024로 변경했다. VGGT 깊이·카메라·마스크·ratio·처리격자 기준 epipolar 허용값은 유지했다. GT와 raw depth는 읽지 않았다.', '', '| 후보 | P1 FIT/보류 | P1a FIT/보류 | 면적 비율 | 시차(°) | 판정 |', '|---|---|---|---:|---:|---|']
 for r in summaries:lines.append(f"| {r['candidate']} | {r['baseline_counts']} | {r['P1a_counts']} | {r['FIT_hull_area_fraction']:.3f} | {r['median_FIT_parallax_deg']} | {r['verdict']} |")
 lines+=['','고해상도에서 새로 추출한 track의 hash 기반 split이며 기존 track과 일대일 동일한 보류 집합은 아니다. 따라서 지원 수 진단이며 성능 개선 비교로 해석하지 않는다. 호모그래피 전이는 FIT track만 사용한 후보이며 다른 시점의 평면 마스크는 사용자 검토 전이다. 최소 지원 기준을 낮추지 않았다. P2 실행 조건이 충족되지 않은 영역은 보정을 수행하지 않는다.', '',f'산출물: `results/planar_residual_depth/{rid}/`.']
 report.write_text('\n'.join(lines)+'\n')
 with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write(f'\n- P1a `{rid}`: M01 승인 동결·원본 RGB 매칭 지원 진단 — [보고서](P1a/{rid}/report.md)\n')
 print(json.dumps({'run_id':rid,'support':summaries},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
