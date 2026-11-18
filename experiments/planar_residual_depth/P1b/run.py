"""Frozen-mask baseline diagnosis only. GT never enters scale or raw-plane fit."""
import json,sys,hashlib,shutil,time
from pathlib import Path
import numpy as np,cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def pfm(path):
 with open(path,'rb') as f:
  assert f.readline().strip()==b'Pf'
  line=f.readline()
  while line.startswith(b'#'):line=f.readline()
  w,h=map(int,line.split());scale=float(f.readline());arr=np.fromfile(f,'<f4' if scale<0 else '>f4').reshape(h,w)
 # Middlebury disparity pixels: ignore magnitude; sign controls endianness.
 return np.flipud(arr)
def plane(X):
 w=np.ones(len(X))
 for _ in range(10):
  c=np.average(X,axis=0,weights=w);_,_,v=np.linalg.svd((X-c)*np.sqrt(w[:,None]),full_matrices=False);n=v[-1];e=-n@c
  r=abs(X@n+e);sigma=max(1e-8,1.4826*np.median(abs(r-np.median(r))));w=np.minimum(1,1.345*sigma/np.maximum(r,1e-12))
 return n,e

def main():
 seq=1;pre='baseline-direction-P1b-'+time.strftime('%Y%m%d')+'-'
 while (BASE/(pre+f'{seq:03}')).exists():seq+=1
 rid=pre+f'{seq:03}';out=BASE/rid;out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/run.py')
 parent=BASE/'P1a-approved-planes-20261117-001';cs=json.loads((parent/'artifacts/approved_candidates.json').read_text());rows=[];fig,axs=plt.subplots(2,4,figsize=(13,7),layout='constrained')
 for index,cid in enumerate(['K01','M01']):
  c=next(c for c in cs if c['id']==cid);sample='000000' if cid=='K01' else 'Adirondack'
  entry=next(x for x in json.loads((BASE/c['source_run']/'samples.json').read_text()) if x['sample']==sample);ims=entry['images'];view=ims[0];K=np.array(view['K']);A=np.array(view['A'])
  rawrid='P1-kitti2015-first20-raw-20261117-001' if cid=='K01' else 'P1-middlebury-raw-20261117-001'
  pp=BASE/rawrid/'artifacts/raw'/sample/'predictions.npz';raw=np.load(pp)['depth'][0,0,...,0];h,w=raw.shape
  yy,xx=np.mgrid[:h,:w];uv=np.stack([xx,yy,np.ones_like(xx)],-1);rays=uv@np.linalg.inv(K).T
  mask=np.array(Image.open(BASE/'review/P1-candidates-v004'/f'{cid}_mask.png'))>0
  # Metric scale from all scene FIT RGB tracks, not plane GT or heldout.
  ts=json.loads((parent/'artifacts/tracks'/f'{sample}.json').read_text());ratios=[];ztri=[]
  for t in ts:
   if t['split']!='fit':continue
   obs={o['view']:o['uv'] for o in t['observations']}
   if 0 not in obs or 1 not in obs:continue
   norm=[];Ps=[]
   for j in [0,1]:
    v=ims[j];R=np.array(v['R']);C=np.array(v['C']);u=np.linalg.inv(np.array(v['K']))@np.r_[obs[j],1.];norm.append(u[:2,None]);Ps.append(np.c_[R,-R@C])
   X=cv2.triangulatePoints(Ps[0],Ps[1],norm[0],norm[1]).ravel();X=X[:3]/X[3]
   Z=[(np.array(ims[j]['R'])@(X-np.array(ims[j]['C'])))[2] for j in [0,1]]
   if min(Z)<=0:continue
   x,y=obs[0];v=float(cv2.getRectSubPix(raw.astype(np.float32),(1,1),(float(x),float(y)))[0,0])
   ratios.append(Z[0]/v);ztri.append(Z[0])
  assert len(ratios)>=12
  scale=float(np.median(ratios));baseline=raw*scale
  # Evaluate GT only after RGB masks and metric scale have been frozen.
  src=ROOT/view['source'];calK=np.linalg.inv(A)@K;fx=calK[0,0]
  B=float(np.linalg.norm(np.array(ims[1]['C'])-np.array(ims[0]['C'])))
  if cid=='M01':
   gtpath=src.parent/'disp0GT.pfm';disp=pfm(gtpath);independent=np.fromfile(ROOT/'data/planar_residual_depth/middlebury/sdk_validation/Adirondack.raw',dtype=np.float32).reshape(disp.shape);assert np.array_equal(disp,independent);cal=json.loads((BASE/c['source_run']/'artifacts/inputs/Adirondack/calibration.json').read_text());doffs=cal['doffs']
  else:
   gtpath=ROOT/'data/planar_residual_depth/kitti2015/source/training/disp_occ_0/000000_10.png';disp=cv2.imread(str(gtpath),cv2.IMREAD_UNCHANGED).astype(float)/256;doffs=(np.linalg.inv(np.array(ims[1]['A']))@np.array(ims[1]['K']))[0,2]-calK[0,2]
  valid=np.isfinite(disp)&(disp>0)&(disp+doffs>0);depth=np.full_like(disp,np.nan);depth[valid]=fx*B/(disp[valid]+doffs)
  orig=uv@np.linalg.inv(A).T;mx=orig[...,0].astype(np.float32);my=orig[...,1].astype(np.float32)
  gt=cv2.remap(depth.astype(np.float32),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))
  evaluation=mask&np.isfinite(gt)&(gt>0);assert evaluation.sum()>10
  # Direct original projection and inverse map agree under provided camera model.
  X=rays[evaluation]*gt[evaluation,None];proj=X@calK.T;projected=proj[:,:2]/proj[:,2:]
  roundtrip=float(np.max(abs(projected-orig[evaluation,:2])));assert roundtrip<1e-8
  n,e=plane(rays[mask]*baseline[mask,None]);ng,eg=plane(X)
  planeDepth=-e/(rays@n);assert np.all(planeDepth[mask]>0)
  b1=baseline.copy();b1[mask]=planeDepth[mask]
  normal=float(np.degrees(np.arccos(np.clip(abs(n@ng),0,1))))
  gr=X@ng+eg;boot=[];rng=np.random.default_rng(0)
  for _ in range(50):
   nb,_=plane(X[rng.integers(0,len(X),len(X))]);boot.append(float(np.degrees(np.arccos(np.clip(abs(nb@ng),0,1)))))
  record={'candidate':cid,'scale_FIT_RGB_only':scale,'scale_FIT_tracks':len(ratios),'GT_valid_pixels':int(evaluation.sum()),
   'normal_error_deg_B0_B1':normal,'GT_plane_residual_RMSE_m':float(np.sqrt(np.mean(gr**2))),
   'GT_bootstrap_normal_p95_deg':float(np.percentile(boot,95)), 'GT_grid_ray_roundtrip_max_px':roundtrip,
   'B0_MAE_m':float(np.mean(abs(baseline[evaluation]-gt[evaluation]))),'B1_MAE_m':float(np.mean(abs(b1[evaluation]-gt[evaluation]))),
   'B2_B3_status':'NOT_RUN_INSUFFICIENT_OBSERVATION_SUPPORT','GT_plane_role':'evaluation only; no solution selection',
   'raw_NPZ_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'GT_sha256':hashlib.sha256(gtpath.read_bytes()).hexdigest()}
  d=out/'artifacts'/cid;d.mkdir(parents=True);np.savez_compressed(d/'arrays.npz',raw_depth=raw,scale=scale,baseline=baseline,B1=b1,residual_B1=b1-baseline,GT=gt,mask=mask,evaluation_mask=evaluation)
  dump(d/'metrics.json',record);rows.append(record)
  for col,(label,img) in enumerate([('RGB',np.array(Image.open(ROOT/view['processed']))),('GT depth',gt),('B0 error (cm)',abs(baseline-gt)*100),('B1 error (cm)',abs(b1-gt)*100)]):
   ax=axs[index,col]
   if col==0:ax.imshow(img)
   else:
    plot=ax.imshow(np.where(evaluation,img,np.nan),cmap='viridis' if col==1 else 'magma',vmin=None if col==1 else 0,vmax=None if col==1 else max(record['B0_MAE_m'],record['B1_MAE_m'])*300);fig.colorbar(plot,ax=ax,shrink=.7)
   poly=c['polygon'];ax.plot([p[0] for p in poly]+[poly[0][0]],[p[1] for p in poly]+[poly[0][1]],color='red',lw=1);ax.set_title(cid+': '+label);ax.axis('off')
 fig.savefig(out/'baseline_GT_comparison.png',dpi=140);plt.close(fig)
 dump(out/'evaluation/summary.json',rows);dump(out/'manifest.json',{'run_id':rid,'stage':'P1b','status':'BASELINE_DIAGNOSIS_ONLY','B2_B3_executed':False,'parent':parent.name})
 report=ROOT/'docs/experiments/planar_residual_depth/P1b'/rid/'report.md';report.parent.mkdir(parents=True,exist_ok=True)
 lines=['# P1b 승인 마스크의 원본 방향·평면화 진단','', '주 보정 B2/B3는 관측 지원 부족으로 미실행이다. GT를 평가에만 읽어 원본 방향·B1 평면화의 영향을 진단했다. scale은 장면 전체 FIT RGB track의 삼각측량 깊이/원본 깊이 median으로 고정했다. 다른 시점 평면 마스크를 확정한 주 실험이나 독립 보류 정확도 평가가 아니다.', '', '| 후보 | raw 법선 오차(°) | GT plane RMSE(mm) | bootstrap 방향 p95(°) | B0 MAE(cm) | B1 MAE(cm) |', '|---|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['candidate']} | {r['normal_error_deg_B0_B1']:.4f} | {r['GT_plane_residual_RMSE_m']*1000:.3f} | {r['GT_bootstrap_normal_p95_deg']:.4f} | {r['B0_MAE_m']*100:.3f} | {r['B1_MAE_m']*100:.3f} |")
 lines+=['', 'Middlebury PFM은 공식 규약에 따라 scale 절댓값을 무시한다. 공식 SDK의 수정하지 않은 ReadFilePFM C++ 함수 출력과 전체 배열 동일성을 검사했다. OpenCV reader는 scale magnitude를 적용하므로 Middlebury 시차의 독립 기준으로 사용하지 않았다. 이전 -002의 Middlebury 수치는 잘못된 scale 적용으로 무효이며 보존만 한다. [공식 규약](https://vision.middlebury.edu/stereo/submit3/upload-format.html)', '', 'GT 평면은10회 robust weighted SVD로 적합했다. bootstrap50회는 지점 재표본의 조건부 안정성이며 센서·등록의 체계적 오차는 포함하지 않는다. Middlebury/KITTI는 원본 GT 격자에 역변환하여 최근접 유효값을 읽었다. metric depth는 fx*baseline/(disparity+cx_right-cx_left)이며 baseline은 m다. GT-camera 광선 왕복검사는 좌표 변환의 대수적 일치 검사이며 외부 센서 정합 검증은 아니다. ETH3D는 distortion/depth 격자 검증 미완료로 평가하지 않았다.', '', '이 단계는 기존 실패 원인을 확정하지 않는다. GT 평면을 solver에 넣거나 raw 방향을 GT에 맞춰 수정하지 않았다. 같은 source-view 마스크의 유효GT 전체를 평가했으며 test holdout 주장 없이 진단으로 제한한다.']
 report.write_text('\n'.join(lines)+'\n')
 with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write(f'\n- P1b `{rid}`: 원본 방향·B1 진단, B2/B3 미실행 — [보고서](P1b/{rid}/report.md)\n')
 print(json.dumps({'run_id':rid,'metrics':rows},indent=2))
if __name__=='__main__':main()
