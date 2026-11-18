"""GT-assisted correspondence diagnosis only; never re-fit, select, or edit planes/tracks."""
import json,hashlib,time,shutil,sys,platform
from pathlib import Path
import numpy as np,cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def affine(x,A):
 p=np.array(A)@np.r_[x,1.];return p[:2]/p[2]
def stats(v):
 x=np.asarray(v,float)
 if not len(x):return {'n':0}
 return {'n':len(x),'mean':float(np.mean(x)),'median':float(np.median(x)),'mean_abs':float(np.mean(abs(x))),
 'p95_abs':float(np.percentile(abs(x),95)),'min':float(np.min(x)),'max':float(np.max(x))}
def sample_gt(d,uv):
 x,y=uv;h,w=d.shape;ix,iy=int(np.floor(x)),int(np.floor(y));nx,ny=int(np.floor(x+.5)),int(np.floor(y+.5))
 if not(0<=nx<w and 0<=ny<h):return {'grade':'outside'}
 nearest=float(d[ny,nx]);result={'grade':'missing','nearest':nearest,'nearest_pixel':[nx,ny]}
 if nearest>0 and np.isfinite(nearest):result['grade']='nearest_only';result['disparity']=nearest
 if 0<=ix<w-1 and 0<=iy<h-1:
  block=d[iy:iy+2,ix:ix+2]
  if (block>0).all() and np.isfinite(block).all():
   fx,fy=x-ix,y-iy;val=(1-fy)*((1-fx)*block[0,0]+fx*block[0,1])+fy*((1-fx)*block[1,0]+fx*block[1,1])
   result.update(grade='bilinear_4_valid',disparity=float(val),local_disparity_min=float(block.min()),local_disparity_max=float(block.max()))
 return result

def main():
 parent=BASE/'P2-kitti-000000-20261117-001';inputs=json.loads((parent/'artifacts/inputs.json').read_text());views=inputs['views'];ts=inputs['tracks']
 original_hashes={str(p.relative_to(parent)):sha(p) for p in parent.rglob('*') if p.is_file()}
 pre='kitti-000000-P2b-'+time.strftime('%Y%m%d')+'-';seq=1
 while (BASE/(pre+f'{seq:03}')).exists():seq+=1
 rid=pre+f'{seq:03}';out=BASE/rid;out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/run.py')
 config={'scope':'diagnosis only on frozen68 LK tracks; no plane reoptimization or GT-based match filtering',
 'primary_GT_support':'disp_noc_0; all4 neighboring disparity values positive -> bilinear at continuous source position',
 'secondary_GT_support':'nearest measured disparity, no invalid-hole filling; sampling error possible',
 'processed_error_flags_px':[.5,1.],'spatial_groups':'fixed mid-x and mid-y of approved source polygon bounding box',
 'all_tracks':68,'FIT':47,'heldout':21,'source_parent':parent.name,'GT_for_optimization':False,
 'interpolation_scope':'only measured four-pixel neighborhood; not GT hole filling',
 'hypothesis':'spatially varying LK correspondence error may be consistent with plane orientation bias; not a causal proof'}
 dump(out/'config_resolved.json',config)
 A0=np.array(views[0]['A']);A1=np.array(views[1]['A']);K0=np.array(views[0]['K']);K1=np.array(views[1]['K']);R0=np.array(views[0]['R']);R1=np.array(views[1]['R']);C0=np.array(views[0]['C']);C1=np.array(views[1]['C'])
 Ko=np.linalg.inv(A0)@K0;K1o=np.linalg.inv(A1)@K1
 assert np.allclose(Ko,K1o) and np.allclose(R0,R1) and np.allclose(R0,np.eye(3)) and np.linalg.norm(C0)<1e-9
 B=float(C1[0]-C0[0]);By=float(C1[1]-C0[1]);Bz=float(C1[2]-C0[2]);assert B>0
 paths={name:ROOT/f'data/planar_residual_depth/kitti2015/source/training/disp_{name}_0/000000_10.png' for name in ['occ','noc']}
 disparities={name:cv2.imread(str(p),cv2.IMREAD_UNCHANGED).astype(float)/256 for name,p in paths.items()}
 arms={a:np.load(parent/'artifacts/arms'/a/'depths.npz') for a in ['B0','B1','B2','B3']}
 poly=np.array(inputs['polygons'][0]);mid=(poly.min(0)+poly.max(0))/2
 rows=[]
 for t in ts:
  obs={o['view']:np.array(o['uv']) for o in t['observations']};u,v=obs[0],obs[1];uo=affine(u,np.linalg.inv(A0));vo=affine(v,np.linalg.inv(A1))
  row={'track_id':t['track_id'],'split':t['split'],'uv_source':u.tolist(),'uv_target_LK':v.tolist(),
   'disparity_LK_original_px':float(uo[0]-vo[0]),'vertical_difference_original_px':float(vo[1]-uo[1]),
   'horizontal_group':'left' if u[0]<mid[0] else 'right','vertical_group':'upper' if u[1]<mid[1] else 'lower','GT':{}}
  for name,d in disparities.items():
   s=sample_gt(d,uo)
   if 'disparity' in s:
    # Exact inversion of horizontal disparity with nonzero y/z baseline.
    rx=(uo[0]-Ko[0,2])/Ko[0,0]
    Z=Ko[0,0]*(B-rx*Bz)/s['disparity']+Bz
    ray=R0.T@np.linalg.inv(K0)@np.r_[u,1.];X=C0+Z*ray;h=K1@R1@(X-C1);expected=h[:2]/h[2];err=v-expected
    target_original=affine(expected,np.linalg.inv(A1));assert abs((uo[0]-target_original[0])-s['disparity'])<1e-8
    s['previous_depth_approximation_minus_exact_m']=float(Ko[0,0]*np.linalg.norm(C1-C0)/s['disparity']-Z)
    s.update(expected_target=expected.tolist(),LK_error_dx_processed_px=float(err[0]),LK_error_dy_processed_px=float(err[1]),LK_error_norm_processed_px=float(np.linalg.norm(err)),GT_depth_m=float(Z),models={})
    for arm,a in arms.items():
     if arm=='B0':z=float(cv2.getRectSubPix(a['corrected_depth'][0].astype(np.float32),(1,1),(float(u[0]),float(u[1])))[0,0])
     else:
      p=a['p'];z=-(1+C0@p)/(ray@p)
     xx=C0+z*ray;hh=K1@R1@(xx-C1);prediction=hh[:2]/hh[2]
     s['models'][arm]={'model_vs_GT_correspondence_px':float(np.linalg.norm(prediction-expected)),
       'model_vs_LK_correspondence_px':float(np.linalg.norm(prediction-v)), 'model_depth_error_m':float(z-Z)}
   row['GT'][name]=s
  rows.append(row)
 def subset(split,grade='bilinear_4_valid',group=None):
  return [r for r in rows if (split=='all' or r['split']==split) and r['GT']['noc']['grade']==grade and (group is None or r[group[0]]==group[1])]
 summaries={}
 for split in ['all','fit','heldout']:
  rr=subset(split);ss=[r['GT']['noc'] for r in rr]
  result={'strict_nonoccluded_count':len(ss),'LK_dx_processed_px':stats([s['LK_error_dx_processed_px'] for s in ss]),
   'LK_dy_processed_px':stats([s['LK_error_dy_processed_px'] for s in ss]),'LK_norm_processed_px':stats([s['LK_error_norm_processed_px'] for s in ss]),
   'over_0_5_px':sum(s['LK_error_norm_processed_px']>.5 for s in ss),'over_1_px':sum(s['LK_error_norm_processed_px']>1 for s in ss),'models':{}}
  for arm in arms:
   result['models'][arm]={'mean_distance_to_GT_correspondence_px':stats([s['models'][arm]['model_vs_GT_correspondence_px'] for s in ss]),
     'mean_distance_to_LK_correspondence_px':stats([s['models'][arm]['model_vs_LK_correspondence_px'] for s in ss])}
  summaries[split]=result
 spatial=[]
 for split in ['all','fit','heldout']:
  for attr in ['horizontal_group','vertical_group']:
   for group in (['left','right'] if attr=='horizontal_group' else ['upper','lower']):
    rr=subset(split,group=(attr,group));dx=[r['GT']['noc']['LK_error_dx_processed_px'] for r in rr]
    spatial.append({'split':split,'axis':attr,'group':group,'dx_processed_px':stats(dx)})
 # Diagnostic linear trend only; this regression is never used to correct tracks/planes.
 rr=subset('fit');trend={'status':'INSUFFICIENT_REFERENCE_SUPPORT'}
 if len(rr)>=6:
  X=np.array([[1,(r['uv_source'][0]-mid[0])/512,(r['uv_source'][1]-mid[1])/256] for r in rr]);y=np.array([r['GT']['noc']['LK_error_dx_processed_px'] for r in rr]);coef,_,rank,_=np.linalg.lstsq(X,y,rcond=None)
  trend={'status':'DIAGNOSTIC_ONLY','n':len(rr),'rank':int(rank),'intercept_px':float(coef[0]),'x_slope_per_512px':float(coef[1]),'y_slope_per_256px':float(coef[2]),'residual_RMSE_px':float(np.sqrt(np.mean((X@coef-y)**2)))}
 coverage={name:{split:{grade:sum(r['split']==split and r['GT'][name]['grade']==grade for r in rows) for grade in ['bilinear_4_valid','nearest_only','missing','outside']} for split in ['fit','heldout']} for name in ['occ','noc']}
 dump(out/'evaluation/track_diagnostics.json',rows);dump(out/'evaluation/summary.json',{'coverage':coverage,'strict_noc_summaries':summaries,'spatial_dx':spatial,'FIT_error_trend':trend})
 fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');rgb=np.array(Image.open(ROOT/views[0]['processed']));axes[0].imshow(rgb)
 allstrict=subset('all');missing=[r for r in rows if r['GT']['noc']['grade']!='bilinear_4_valid']
 if missing:axes[0].scatter([r['uv_source'][0] for r in missing],[r['uv_source'][1] for r in missing],c='gray',marker='x',s=18,label='GT unsupported (strict)')
 if allstrict:
  xy=np.array([r['uv_source'] for r in allstrict]);values=np.array([r['GT']['noc']['LK_error_dx_processed_px'] for r in allstrict]);im=axes[0].scatter(xy[:,0],xy[:,1],c=values,cmap='RdBu_r',vmin=-1,vmax=1,s=24,edgecolors='white',linewidths=.3);fig.colorbar(im,ax=axes[0],label='LK x - GT-expected x (processed px)',shrink=.7,extend='both')
 axes[0].set_title('Frozen source tracks: GT-supported horizontal error');axes[0].axis('off')
 for split,marker,col in [('fit','o','tab:blue'),('heldout','x','tab:orange')]:
  rr=subset(split);axes[1].scatter([r['uv_source'][0] for r in rr],[r['GT']['noc']['LK_error_dx_processed_px'] for r in rr],marker=marker,c=col,label=split)
 axes[1].axhline(0,c='black',lw=1);axes[1].axhline(.5,c='gray',ls='--');axes[1].axhline(-.5,c='gray',ls='--');axes[1].set(xlabel='Source u (processed px)',ylabel='LK x - GT-expected x (px)',title='Same plane: error depends on position?');axes[1].legend();axes[1].grid(alpha=.2)
 fig.suptitle('P2b diagnosis only: strict non-occluded GT, 4 valid neighbors; no tracks changed')
 fig.savefig(out/'correspondence_error_map.png',dpi=150);plt.close(fig)
 # Show every available strict GT comparison; no best/worst selection.
 fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 for ax,split in zip(axes,['fit','heldout']):
  vals=[summaries[split]['models'][arm]['mean_distance_to_GT_correspondence_px'].get('mean',np.nan) for arm in arms]
  ax.bar(list(arms),vals,color=['gray','silver','tab:blue','tab:orange']);ax.set(title=f'{split}: GT supported n={summaries[split]["strict_nonoccluded_count"]}',ylabel='Mean model-to-GT correspondence distance (px)')
 fig.suptitle('Stored B0–B3 compared to GT-expected target positions; no optimizer rerun');fig.savefig(out/'model_vs_GT_correspondences.png',dpi=150);plt.close(fig)
 unchanged=all(sha(parent/p)==h for p,h in original_hashes.items());assert unchanged
 dump(out/'manifest.json',{'run_id':rid,'stage':'P2b','status':'DIAGNOSTIC_COMPLETED','parent':parent.name,'parent_hashes':original_hashes,'parent_unchanged':unchanged,
 'GT_used_for_diagnosis':True,'GT_used_for_optimization':False,'tracks_changed':False,'planes_changed':False,'GT_sources_sha256':{k:sha(p) for k,p in paths.items()}})
 dump(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'opencv':cv2.__version__,'platform':platform.platform(),'execution':'CPU only'})
 report=ROOT/'docs/experiments/planar_residual_depth/P2b'/rid/'report.md';report.parent.mkdir(parents=True,exist_ok=True)
 lines=['# P2b K01 대응점과 GT 기하 차이 진단','', 'K01의 기존 FIT47개·보류21개와 B0–B3를 고정하고 GT가 예측하는 오른쪽 위치와 비교했다. GT는 진단에만 사용했으며 대응점 필터·마스크·scale·평면을 수정하거나 optimizer를 재실행하지 않았다.','',
 '## 참조와 유효성','', 'KITTI 기준시각 왼쪽 disp_noc_0를 주 참조로 사용했다. source의 subpixel 위치 주변4개 값이 모두 유효할 때만 bilinear 시차를 계산했다. 결측을 보간해 채우지 않았다. nearest-only는 별도 보조 목록에 남기고 주 통계에서 제외했다. disp_occ_0의 지원 상태도 따로 기록했다. 수평 GT 시차를 제공 카메라의 y/z baseline 성분까지 포함해 역산했다. GT 수평 시차와 전체카메라 재투영 시차의1e-8px 일치를 검사했다. 수직 기대 위치는 별도 센서 GT가 아니라 제공된 calibration에서 계산한다. 기존 f*Euclidean-baseline/disparity 깊이와의 작은 차이는 track별로 기록했다.', '',
 '| split | 원래 track | 엄격 GT 지원 | LK 위치오차 평균(px) | 수평오차 평균(px) | 0.5px 초과 | 1px 초과 |','|---|---:|---:|---:|---:|---:|---:|']
 for split,n in [('fit',47),('heldout',21)]:
  s=summaries[split];lines.append(f"| {split} | {n} | {s['strict_nonoccluded_count']} | {s['LK_norm_processed_px'].get('mean')} | {s['LK_dx_processed_px'].get('mean')} | {s['over_0_5_px']} | {s['over_1_px']} |")
 lines+=['','## 해석 제한','', '비교의 기준은 센서/모델로 구성된 KITTI GT이며 완전한 무오차 대응 좌표가 아니다. bilinear4-valid 조건도 GT의 양자화·캘리브레이션·센서·경계 오차를 제거하지 못한다. 참조가 없는 track을 잘못된 매칭이라고 판정하지 않는다. 이 결과는 이전에 본 개발 ROI의 진단이며 GT로 매칭을 선별한 새 알고리즘 성능이 아니다.', '',
 'source mask 내부의 LK 패치에는 경계/레일/반사/노면 굴곡의 영향이 있을 수 있다. 공간적 오차 추세가 보여도 그것만으로 특정 물체·조명·카메라 오차를 원인으로 확정하지 않는다. GT 참조 자체가 비평면 노면을 포함하므로 모든 대응이 단일 평면에 완전히 일치할 필요는 없다.', '',
 '각 track의 occ/noc 지원·기대 좌표·오차, 공간 그룹 통계, 모델별 LK/GT 대응 거리는 경량 JSON에 저장했다. FIT 오차의 선형회귀는 설명용이며 보정에 사용하지 않는다. 기존 FIT 입력·출력 전체 hash가 그대로인지 확인했다.', '',f'결과: `results/planar_residual_depth/{rid}/`.']
 report.write_text('\n'.join(lines)+'\n')
 for name in ['summary.json','track_diagnostics.json']:shutil.copy2(out/'evaluation'/name,report.parent/name)
 with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write(f'\n- P2b `{rid}`: 고정 대응점의 GT 기하 진단 — [보고서](P2b/{rid}/report.md)\n')
 print(json.dumps({'run_id':rid,'coverage':coverage,'summaries':summaries,'spatial':spatial,'trend':trend,'report':str(report)},indent=2))
if __name__=='__main__':main()
