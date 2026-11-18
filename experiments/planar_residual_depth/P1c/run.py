"""RGB-only Shi-Tomasi + pyramidal LK diagnostic on frozen planar source masks."""
import json,time,shutil,hashlib
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def transform(uv,T):
 x=np.c_[np.asarray(uv).reshape(-1,2),np.ones(len(np.asarray(uv).reshape(-1,2)))]@T.T
 return x[:,:2]/x[:,2:]
def main():
 seq=1;prefix='approved-planes-P1c-'+time.strftime('%Y%m%d')+'-'
 while (BASE/(prefix+f'{seq:03}')).exists():seq+=1
 rid=prefix+f'{seq:03}';out=BASE/rid;out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/run.py')
 parent=BASE/'P1a-approved-planes-20261117-001';cs=json.loads((parent/'artifacts/approved_candidates.json').read_text())
 config={'max_corners':1000,'quality_level':.01,'min_distance_native_px':3,'block_size':3,'LK_window':[15,15],'LK_pyramid_levels':4,
 'LK_iterations':30,'LK_epsilon':.01,'LK_min_eigen':.0001,'FB_threshold_processed_px':.5,'epipolar_threshold_processed_px':1.,
 'FIT_min':12,'heldout_min':6,'FIT_hull_area_fraction_min':.1,'median_FIT_parallax_deg_min':1.,'seed':0,
 'matching_images':'same original RGB capped at1024 from P1a','GT_read':False,'raw_depth_read':False,'target_mask':'not yet human approved; all tracks provisional'}
 dump(out/'config_resolved.json',config);summary=[]
 for c in cs:
  sample='Adirondack' if c['id']=='M01' else '000000' if c['id']=='K01' else 'pipes'
  entry=next(x for x in json.loads((BASE/c['source_run']/'samples.json').read_text()) if x['sample']==sample);views=entry['images']
  gray=[];Ts=[]
  for i,v in enumerate(views):
   path=parent/'artifacts/matching_rgb'/sample/f'view{i}.png';im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE);gray.append(im)
   source=Image.open(ROOT/v['source']);ow,oh=source.size;cw=min(ow,round(oh*2));ch=min(oh,round(ow*2));l=(ow-cw)//2;t=(oh-ch)//2;nh,nw=im.shape
   An=np.array([[nw/cw,0,(-l+.5)*nw/cw-.5],[0,nh/ch,(-t+.5)*nh/ch-.5],[0,0,1.]])
   Ts.append(np.array(v['A'])@np.linalg.inv(An))
  native_poly=transform(c['polygon'],np.linalg.inv(Ts[0]));mask=np.zeros_like(gray[0]);cv2.fillPoly(mask,[np.round(native_poly).astype(np.int32)],255)
  corners=cv2.goodFeaturesToTrack(gray[0],1000,.01,3,mask=mask,blockSize=3)
  records=[];reasons={};accepted_by_source={};stats=[]
  if corners is not None:
   p0=corners.astype(np.float32);uv0=transform(p0,Ts[0])
   inside=np.array([cv2.pointPolygonTest(np.array(c['polygon'],np.float32),tuple(uv),False)>=0 for uv in uv0]);p0=p0[inside];uv0=uv0[inside]
   for j in range(1,len(views)):
    params=dict(winSize=(15,15),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
    pj,sf,ef=cv2.calcOpticalFlowPyrLK(gray[0],gray[j],p0,None,**params)
    if pj is None:continue
    pb,sb,eb=cv2.calcOpticalFlowPyrLK(gray[j],gray[0],pj,None,**params)
    if pb is None:continue
    uvj=transform(pj,Ts[j]);back=transform(pb,Ts[0]);fb=np.linalg.norm(back-uv0,axis=1)
    a,b=views[0],views[j];R0,Rj=np.array(a['R']),np.array(b['R']);C0,Cj=np.array(a['C']),np.array(b['C']);K0,Kj=np.array(a['K']),np.array(b['K'])
    t=Rj@(C0-Cj);R=Rj@R0.T;tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    F=np.linalg.inv(Kj).T@tx@R@np.linalg.inv(K0);x=np.c_[uv0,np.ones(len(uv0))];y=np.c_[uvj,np.ones(len(uvj))];ly=x@F.T;lx=y@F
    epi=abs(np.sum(y*ly,axis=1))/np.maximum(1e-12,np.minimum(np.linalg.norm(ly[:,:2],axis=1),np.linalg.norm(lx[:,:2],axis=1)))
    w,h=b['size'];valid=(sf.ravel()==1)&(sb.ravel()==1)&np.isfinite(uvj).all(1)&(fb<=.5)&(epi<=1)&(uvj[:,0]>=0)&(uvj[:,0]<w)&(uvj[:,1]>=0)&(uvj[:,1]<h)
    # Eliminate collapsed tracks landing on same target neighborhood (processed grid).
    target_used=set();kept=[]
    for k in np.argsort(fb):
     if not valid[k]:continue
     cell=tuple(np.floor(uvj[k]/2).astype(int))
     if cell in target_used:continue
     target_used.add(cell);kept.append(int(k))
    for k in kept:
     ra=R0.T@np.linalg.inv(K0)@x[k];rb=Rj.T@np.linalg.inv(Kj)@y[k]
     par=float(np.degrees(np.arccos(np.clip(ra@rb/np.linalg.norm(ra)/np.linalg.norm(rb),-1,1))))
     accepted_by_source.setdefault(k,[]).append({'view':j,'uv':uvj[k].tolist(),'fb_px':float(fb[k]),'epipolar_px':float(epi[k]),'parallax_deg':par})
    stats.append({'target_view':j,'seeds':len(p0),'LK_forward_passed':int(sf.sum()),'LK_backward_passed':int(sb.sum()),'gated':len(kept)})
   for k,other in sorted(accepted_by_source.items()):
    tid=hashlib.sha256((c['id']+':'+','.join(f'{v:.4f}' for v in uv0[k])).encode()).hexdigest();bucket=int(hashlib.sha256(('0:'+tid).encode()).hexdigest()[:8],16)%10
    records.append({'track_id':tid,'split':'heldout' if bucket<3 else 'fit','observations':[{'view':0,'uv':uv0[k].tolist()}]+other})
  counts={s:sum(t['split']==s for t in records) for s in ['fit','heldout']}
  fit=np.array([t['observations'][0]['uv'] for t in records if t['split']=='fit'],np.float32)
  area=float(cv2.contourArea(cv2.convexHull(fit))/cv2.contourArea(np.array(c['polygon'],np.float32))) if len(fit)>=3 else 0.
  angles=[o['parallax_deg'] for t in records if t['split']=='fit' for o in t['observations'][1:]];par=float(np.median(angles)) if angles else None
  eligible=counts['fit']>=12 and counts['heldout']>=6 and area>=.1 and par is not None and par>=1
  row={'candidate':c['id'],'counts':counts,'FIT_hull_area_fraction':area,'median_FIT_parallax_deg':par,'observation_gate_passed':eligible,
    'verdict':'OBSERVATION_GATE_PASSED_TARGET_MASK_REVIEW_PENDING' if eligible else 'INSUFFICIENT_SUPPORT','per_target':stats}
  summary.append(row);dump(out/'artifacts'/c['id']/'tracks.json',records)
  image=Image.open(ROOT/views[0]['processed']).convert('RGB');d=ImageDraw.Draw(image);poly=[tuple(x) for x in c['polygon']];d.line(poly+[poly[0]],fill='red',width=2)
  for t in records:
   x,y=t['observations'][0]['uv'];col='lime' if t['split']=='fit' else 'orange';d.ellipse((x-2,y-2,x+2,y+2),outline=col,width=1)
  image.save(out/'artifacts'/c['id']/'source_tracks.png')
  # Target proposal derived exclusively from FIT tracks; never a human-approved mask.
  for j in range(1,len(views)):
   pairs=[]
   for t in records:
    obs={o['view']:o['uv'] for o in t['observations']}
    if t['split']=='fit' and j in obs:pairs.append((obs[0],obs[j]))
   if len(pairs)<4:continue
   a=np.float32([p[0] for p in pairs]);b=np.float32([p[1] for p in pairs]);H,ins=cv2.findHomography(a,b,cv2.RANSAC,2.)
   if H is None:continue
   polyj=cv2.perspectiveTransform(np.float32(c['polygon'])[None],H)[0]
   dump(out/'artifacts'/c['id']/f'target{j}_proposal.json',{'polygon':polyj.tolist(),'status':'PENDING_HUMAN_REVIEW','FIT_H_inliers':int(ins.sum())})
   target=Image.open(ROOT/views[j]['processed']).convert('RGB');dd=ImageDraw.Draw(target);pol=[tuple(p) for p in polyj];dd.line(pol+[pol[0]],fill='red',width=2)
   for t in records:
    for o in t['observations']:
     if o['view']==j:
      x,y=o['uv'];dd.ellipse((x-2,y-2,x+2,y+2),outline='lime' if t['split']=='fit' else 'orange',width=1)
   target.save(out/'artifacts'/c['id']/f'target{j}_tracks.png')
 dump(out/'evaluation/support.json',summary);dump(out/'manifest.json',{'run_id':rid,'stage':'P1c','status':'COMPLETED_SUPPORT_DIAGNOSTIC','source_mask_version':'P1-candidates-v004 (M01 approved in P1a)','parent':'P1a-approved-planes-20261117-001','GT_read':False,'raw_depth_read':False,'correction_executed':False})
 report=ROOT/'docs/experiments/planar_residual_depth/P1c'/rid/'report.md';report.parent.mkdir(parents=True,exist_ok=True)
 lines=['# P1c 코너 검출·LK 추적 관측 지원 진단','', '학습 모델·GT·raw depth를 사용하지 않고 승인한 source polygon 내부의 Shi–Tomasi 코너를 원본 RGB(긴변 최대1024)에서 추적했다. source→target→source 왕복오차0.5px, 양방향 epipolar1px(기존512처리격자), 영상 내 유효성 및 target2px cell 중복을 검사했다. 설정은 실행 전에 config로 저장했다.','', '| 후보 | FIT | 보류 | 면적 비율 | 시차(°) | 판정 |','|---|---:|---:|---:|---:|---|']
 for r in summary:lines.append(f"| {r['candidate']} | {r['counts']['fit']} | {r['counts']['heldout']} | {r['FIT_hull_area_fraction']:.3f} | {r['median_FIT_parallax_deg']} | {r['verdict']} |")
 lines+=['','FIT/보류는 source 좌표를 기반으로 한 track ID hash/seed0으로 분리했다. 새로운 특징 검출기로 얻은 보류 집합이므로 SIFT 대비 정확도 개선을 직접 주장하지 않는다. 가까운 코너의 패치는 상관될 수 있어 대응 개수를 독립 물리 표본 수로 해석하지 않는다. LK 패치는 평면 경계를 넘을 수 있으며 target mask 검토와 추가 퇴화 검사가 남아 있다. source mask만 사용자 승인 상태이고 target polygon은 FIT 호모그래피로 생성한 검토 후보다. 관측 지원 gate 통과가 실제 깊이 정확도나 동일 표면 소속의 검증은 아니다.', '',f'산출물: `results/planar_residual_depth/{rid}/`.']
 report.write_text('\n'.join(lines)+'\n')
 with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write(f'\n- P1c `{rid}`: LK 관측 지원 — [보고서](P1c/{rid}/report.md)\n')
 print(json.dumps({'run_id':rid,'support':summary},indent=2))
if __name__=='__main__':main()
