"""P9a-d aggregation and image-backed alignment/depth diagnostics, no fitting."""
import sys,json,csv,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).parents[1]/'P5_P8'))
from common import *
from photo import *

ARMS=['C_B2','C_B3','F9C_MATCH_TRUST','P9B_PHOTO_FIXED','P9C_PHOTO_FREE']
def aggregate(rows):
 result={}
 for ds in ['kitti','middlebury']:
  rr=[r for r in rows if r['dataset']==ds];active=[r for r in rr if r['metrics']['P9C_PHOTO_FREE']['status']=='PHOTO_REFINED' and r['metrics']['P9B_PHOTO_FIXED']['status']=='PHOTO_REFINED'];st={'existing_cases':len(rr),'paired_photo_refined_cases':len(active),'fallback_cases':len(rr)-len(active),'all_retained_mean_MAE_cm':{},'paired_active_mean_MAE_cm':{},'paired_active_mean_normal_error_deg':{},'paired_changes':{},'P9a':{},'P9d':{}}
  for a in ARMS:
   st['all_retained_mean_MAE_cm'][a]=float(np.mean([r['metrics'][a]['ROI_MAE_m'] for r in rr])*100);st['paired_active_mean_MAE_cm'][a]=float(np.mean([r['metrics'][a]['ROI_MAE_m'] for r in active])*100) if active else None;st['paired_active_mean_normal_error_deg'][a]=float(np.mean([r['metrics'][a]['normal_error_deg'] for r in active])) if active else None
  for a,b in [('P9B_PHOTO_FIXED','C_B2'),('P9C_PHOTO_FREE','C_B3'),('P9C_PHOTO_FREE','P9B_PHOTO_FIXED'),('P9C_PHOTO_FREE','F9C_MATCH_TRUST')]:
   dd=np.array([100*(r['metrics'][a]['ROI_MAE_m']-r['metrics'][b]['ROI_MAE_m']) for r in active]);rng=np.random.default_rng(0);bb=dd[rng.integers(0,len(dd),(10000,len(dd)))].mean(1) if len(dd) else np.array([np.nan]);st['paired_changes'][a+'_vs_'+b]={'cases':len(dd),'mean_delta_MAE_cm':float(dd.mean()) if len(dd) else None,'nominal_ID_bootstrap_95_cm':np.percentile(bb,[2.5,97.5]).tolist() if len(dd) else None,'lower_depth_MAE_cases':int(np.sum(dd<0)),'lower_heldout_photo_cost_cases':sum(r['metrics'][a]['photometric']['heldout']['Huber_cost']<r['metrics'][b]['photometric']['heldout']['Huber_cost'] for r in active),'joint_lower_depth_and_normal_no_larger_heldout_photo_cases':sum(r['metrics'][a]['ROI_MAE_m']<r['metrics'][b]['ROI_MAE_m'] and r['metrics'][a]['normal_error_deg']<r['metrics'][b]['normal_error_deg']-1e-6 and r['metrics'][a]['photometric']['heldout']['Huber_cost']<=r['metrics'][b]['photometric']['heldout']['Huber_cost'] for r in active)}
  pairs=[]
  for r in active:
   a,b=r['metrics']['C_B2'],r['metrics']['C_B3'];dp=b['photometric']['heldout']['Huber_cost']-a['photometric']['heldout']['Huber_cost'];dd=b['ROI_MAE_m']-a['ROI_MAE_m'];same=a['photo_heldout_GT_valid_pixels']>=12 and b['photo_heldout_GT_valid_pixels']>=12;local=b['photo_heldout_GT_MAE_m']-a['photo_heldout_GT_MAE_m'] if same else None;pairs.append({'sample':r['sample'],'B3_minus_B2_heldout_photo_cost':dp,'B3_minus_B2_ROI_MAE_cm':dd*100,'ordering_agrees':bool(dp*dd>0),'photo_heldout_GT_pixels':a['photo_heldout_GT_valid_pixels'],'same_pixel_GT_ordering_agrees':bool(dp*local>0) if same else None,'oracle_ref_inside_fraction':r['oracle']['heldout']['sample_inside_image_fraction']})
  st['P9a']={'paired_cases':len(pairs),'photo_and_ROI_depth_order_agree_cases':sum(r['ordering_agrees'] for r in pairs),'same_pixel_GT_comparable_cases':sum(r['same_pixel_GT_ordering_agrees'] is not None for r in pairs),'same_pixel_GT_order_agree_cases':sum(r['same_pixel_GT_ordering_agrees'] is True for r in pairs),'oracle_photo_below_B2_cases':sum(r['oracle']['heldout']['Huber_cost']<r['metrics']['C_B2']['photometric']['heldout']['Huber_cost'] for r in active),'oracle_photo_below_B3_cases':sum(r['oracle']['heldout']['Huber_cost']<r['metrics']['C_B3']['photometric']['heldout']['Huber_cost'] for r in active),'case_pairs':pairs}
  for a in ARMS:
   st['P9d'][a]={'active_brightness_bounds_hit':sum(r['metrics'][a]['photometric']['fit']['brightness_bound_hit'] for r in active),'active_warp_trust_boundary_hit':sum(r['metrics'][a]['max_dense_source_warp_delta_from_B2_per_coordinate_px']>=2.9999 for r in active),'mean_heldout_high_residual_fraction':float(np.mean([r['metrics'][a]['photometric']['heldout']['high_residual_fraction_norm_gt_0_03'] for r in active])) if active else None,'mean_heldout_raw_no_affine_RGB_MAE':float(np.mean([r['metrics'][a]['photo_heldout_no_affine']['RGB_mean_abs_error'] for r in active])) if active else None,'mean_heldout_affine_RGB_MAE':float(np.mean([r['metrics'][a]['photometric']['heldout']['RGB_mean_abs_error'] for r in active])) if active else None}
  result[ds]=st
 return result

def render(row,out):
 fit=ROOT/row['fit_dir'];info=json.loads((fit/'case.json').read_text());data=np.load(fit/'inputs.npz');mask=data['masks'][0];views=info['views'];images=[np.array(Image.open(ROOT/v['processed']).convert('RGB'),float)/255 for v in views];parts=np.load(fit/'photo_support.npz');pgt=np.array(row['oracle']['GT_reference_plane_p']);gtdata=np.load(ROOT/info['parent_P4_evaluation']/'GT_arrays.npz');gt=gtdata['GT_depth'];valid=gtdata['evaluation_mask'];h,w=mask.shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()];uu=uv[mask.ravel()];ys,xs=np.where(mask);sl=np.s_[max(0,ys.min()-8):min(h,ys.max()+9),max(0,xs.min()-8):min(w,xs.max()+9)];dest=ROOT/row['evaluation_dir'];poses={a:np.load(fit/'arms'/(a+'.npz')) for a in ARMS};columns=ARMS+['ORACLE_GT_PLANE'];cm=plt.get_cmap('magma').copy();cm.set_bad('#999999')
 fig,axs=plt.subplots(3,7,figsize=(19,8),layout='constrained');axs[0,0].imshow(images[0][sl]);axs[0,0].set_title('Source RGB');split=np.full((h,w),np.nan)
 for name,col in [('fit',0),('heldout',1)]:
  p=parts[name].astype(int)
  if len(p):split[p[:,1],p[:,0]]=col
 axs[1,0].imshow(split[sl],cmap=plt.get_cmap('coolwarm',2),vmin=0,vmax=1);axs[1,0].set_title('Blue FIT / red photo-heldout');axs[2,0].imshow(images[1][sl]);axs[2,0].set_title('Target RGB before warp')
 block=((x//8+y//8)%2).astype(bool);warp_saves={}
 for j,a in enumerate(columns,1):
  if a=='ORACLE_GT_PLANE':p=pgt;g=row['oracle']['gain'];b=row['oracle']['bias']
  else:z=poses[a];p=z['p'];g=float(z['gain']);b=float(z['bias'])
  xy,_=project(p,uu,views);val,grad,inside=bilinear(images[1],xy);warp=np.full((h,w,3),np.nan);warp[mask]=np.clip(g*val+b,0,1);diff=np.full((h,w),np.nan);diff[mask]=np.mean(abs(g*val+b-images[0][mask]),axis=1);blend=np.where(block[:,:,None],images[0],warp);axs[0,j].imshow(warp[sl]);axs[0,j].set_title(a.replace('P9B_PHOTO_FIXED','P9b photo fixed').replace('P9C_PHOTO_FREE','P9c photo free').replace('F9C_MATCH_TRUST','Feature same trust'));axs[1,j].imshow(blend[sl]);axs[1,j].set_title('8px RGB checkerboard');ie=axs[2,j].imshow(diff[sl],cmap=cm,vmin=0,vmax=.1);score=row['oracle']['heldout'] if a=='ORACLE_GT_PLANE' else row['metrics'][a]['photometric']['heldout'];axs[2,j].set_title('photo-heldout RGB MAE\n'+(f"{score['RGB_mean_abs_error']:.4f}" if score else 'not available'));warp_saves[a]=warp
 for ax in axs.ravel():ax.axis('off')
 fig.colorbar(ie,ax=axs[2,1:],label='mean abs RGB error [0,1], common range',shrink=.65);fig.suptitle(row['dataset']+' '+row['sample']+' | oracle is evaluation-only; fixed support, no residual deletion');fig.savefig(dest/'RGB_warp_alignment.png',dpi=130);plt.close(fig)
 # Depth evaluation retains all GT-valid source ROI pixels, not only photometric samples.
 vals=np.concatenate([gt[valid]]+[z['corrected_depth'][0][mask] for z in poses.values()]);lo,hi=np.percentile(vals,[1,99]);bound=max(1e-6,float(np.percentile(np.concatenate([abs(z['corrected_depth'][0][valid]-gt[valid]) for z in poses.values()]),95)));dc=plt.get_cmap('viridis').copy();dc.set_bad('#999999');ec=plt.get_cmap('RdBu_r').copy();ec.set_bad('#999999');fig,axs=plt.subplots(2,6,figsize=(17,6),layout='constrained');axs[0,0].imshow(images[0][sl]);axs[0,0].set_title('Source RGB');axs[1,0].imshow(np.where(valid,gt,np.nan)[sl],cmap=dc,vmin=lo,vmax=hi);axs[1,0].set_title('GT; gray=not evaluated')
 for j,(a,z) in enumerate(poses.items(),1):
  d=z['corrected_depth'][0];im=axs[0,j].imshow(np.where(mask,d,np.nan)[sl],cmap=dc,vmin=lo,vmax=hi);axs[0,j].set_title(a);ie=axs[1,j].imshow(np.where(valid,d-gt,np.nan)[sl],cmap=ec,vmin=-bound,vmax=bound);m=row['metrics'][a];axs[1,j].set_title(f"MAE {100*m['ROI_MAE_m']:.2f} cm\nnormal {m['normal_error_deg']:.2f} deg")
 for ax in axs.ravel():ax.axis('off')
 fig.colorbar(im,ax=axs[0,1:],label='camera-Z m',shrink=.65);fig.colorbar(ie,ax=axs[1,1:],label='prediction - GT m',shrink=.65);fig.suptitle(row['dataset']+' '+row['sample']+' | same ROI/calibration/scale');fig.savefig(dest/'depth_GT_errors.png',dpi=130);plt.close(fig)
 q=rays(uu,views[0]);XYZ=np.array([np.array(views[0]['C'])+z['corrected_depth'][0][mask,None]*q for z in poses.values()]);np.savez_compressed(dest/'ROI_pointmaps.npz',arm_names=np.array(ARMS),X=XYZ,RGB=np.rint(images[0][mask]*255).astype('uint8'),pixel_uv=uu,frame_id=np.array(views[0]['source']),scope=np.array('finite source0 ROI, same provided reference camera coordinates'))


def main():
 out=BASE/sys.argv[1];rows=json.loads((out/'summary.json').read_text());fit=BASE/json.loads((out/'config_resolved.json').read_text())['parent_fit'];stats=aggregate(rows);dump(out/'aggregate.json',stats);shutil.copy2(__file__,out/'source/report.py')
 flat=[]
 for r in rows:
  for a,m in r['metrics'].items():
   flat.append({'dataset':r['dataset'],'sample':r['sample'],'arm':a,'status':m['status'],'reason':m['reason'],'photo_support_passed':r['support']['passed'],'ROI_MAE_cm':100*m['ROI_MAE_m'],'normal_error_deg':m['normal_error_deg'],'photo_heldout_GT_pixels':m['photo_heldout_GT_valid_pixels'],'heldout_Huber_cost':m['photometric']['heldout']['Huber_cost'] if m['photometric']['heldout'] else None,'heldout_RGB_MAE':m['photometric']['heldout']['RGB_mean_abs_error'] if m['photometric']['heldout'] else None,'gain':m['gain'],'bias':m['bias'],'max_warp_delta_px':m['max_dense_source_warp_delta_from_B2_per_coordinate_px']})
 with (out/'per_case_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 if '--summary-only' not in sys.argv:
  for r in rows:render(r,out)
  fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
  for ax,(ds,st) in zip(axs,stats.items()):
   for p in st['P9a']['case_pairs']:ax.scatter(p['B3_minus_B2_heldout_photo_cost'],p['B3_minus_B2_ROI_MAE_cm']);ax.annotate(p['sample'],(p['B3_minus_B2_heldout_photo_cost'],p['B3_minus_B2_ROI_MAE_cm']),fontsize=7)
   ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7);ax.set_title(ds);ax.set_xlabel('B3 - B2 photo-heldout Huber cost');ax.set_ylabel('B3 - B2 GT ROI MAE cm')
  fig.suptitle('P9a: same-sign quadrants agree; no GT used in fitting');fig.savefig(out/'P9a_photo_vs_depth_order.png',dpi=140);plt.close(fig)
 report=ROOT/'docs/experiments/planar_residual_depth/P9'/out.name/'report.md';report.parent.mkdir(parents=True,exist_ok=True)
 lines=['# P9a–P9d: 원본 RGB 직접 정렬에 의한 평면 깊이 보정','', '**기존21개 사례에서 P9a–P9d를 처리했다. RGB 보정은 KITTI13개/Middlebury3개에서 실행됐고, 나머지5개는 영상 블록 지원 부족으로 B2를 유지했다. 이전8개 미지원 사례는 별도 보존했다. 같은 장면을 이미 분석한 후속 탐색이며 독립 검증이 아니다.**','',
 '## 사전 고정 방법과 GT 경계','',
 '모든 규칙을 fit 전에 `protocol.json`으로 고정했다. 기존 source/target 평면 mask, RGB, 제공 K/pose, raw depth, 공통 FIT-RGB scale을 유지했다. P9b/c는 고정 LK 대응점 잔차를 사용하지 않고 source0→target1 평면 warp의 RGB 차이를 최소화했다. RGB[0,1] sRGB에 radial Huber0.03, 전역 scalar gain[0.7,1.3]·bias[−0.15,0.15]를 사용했다. 광도 모델은 실제 카메라 response나 반사 모델을 완전히 설명하지 못한다.',
 '', 'P9b는 기존 B1의 법선 방향 고정·거리만 보정, P9c는 방향·거리 공동 보정이다. Gaussian sigma1.0/0.5/0.0에서 같은 픽셀 support를 사용했다. 양의 깊이와 기존 B2 대비 source polygon corner의 target warp 각 좌표±3px를 제약했다. 최종 dense source ROI에서도 한도를 검증했다. Feature same-trust 대조군은 기존 LK FIT 목적함수를 같은 제약에서 최적화해 탐색 범위 변경 효과를 분리했다.',
 '', 'Source mask4px·target mask9px 침식 후 B2와B3에서 모두 target 내부인 source 점들을 고정했다. 32px 영상 tile의 경계8px를 제외하고 tile hash seed0으로 FIT/보류70/30을 나눴다. FIT64px·2tiles, 보류32px·1tile, FIT hull10% 기준을 낮추지 않았다. 최적화 중 현재 잔차에 따라 픽셀을 삭제하거나 mask를 바꾸지 않았다. photometric 보류는 P9 갱신에 쓰지 않은 픽셀이지만 원래 RGB/LK 기반 초기값이 이미 같은 관측을 반영했을 수 있어 독립 테스트로 부르지 않는다.',
 '', 'GT는 모든 fit 및 pre-GT 기술 검사가 끝난 뒤 별도 프로세스에서 읽었다. GT 참조 평면의 warp는 사후 oracle 진단만 하며 초기화·support·해 선택에 전달하지 않았다. Oracle의 gain/bias만 고정된 FIT 픽셀에서 적합했다. GT 참조 평면도 센서 GT의 근사 평면이며 실제 노면 굴곡·참조 불확실성은 기존 P4 분석과 함께 해석한다.',
 '', '## P9a: 영상 정렬 오차는 깊이 오차와 연결되는가?','', '| 데이터셋 | 보정 지원 사례 | B2/B3의 RGB 보류 순위와 ROI GT 깊이 순위 일치 | 같은 RGB 보류 위치의 GT로 비교 | GT참조 plane의 RGB 보류 손실이 B2보다 작은 사례 |','|---|---:|---:|---:|---:|']
 for ds,s in stats.items():
  a=s['P9a'];lines.append(f"| {ds} | {s['paired_photo_refined_cases']} | {a['photo_and_ROI_depth_order_agree_cases']}/{a['paired_cases']} | {a['same_pixel_GT_order_agree_cases']}/{a['same_pixel_GT_comparable_cases']} | {a['oracle_photo_below_B2_cases']}/{a['paired_cases']} |")
 lines+=['','대부분의 지원 사례에서 정렬 점수와 깊이 순위가 일치했지만 예외가 있다. GT 참조 평면이 항상 RGB 오차를 최소화하지도 않는다. 밝기·가림·반사·무늬·GT 평면 근사·카메라 보정 잔차·리샘플링 오차 등은 가능한 요인이며 이번 관측만으로 개별 원인을 확정하지 않는다.','',
 '## P9b/c: 동일 지원 사례의 paired 정확도','', '| 방법 | KITTI 평균 MAE(cm),13개 | Middlebury 평균 MAE(cm),3개 |','|---|---:|---:|']
 for a in ARMS:lines.append(f"| {a} | {stats['kitti']['paired_active_mean_MAE_cm'][a]:.3f} | {stats['middlebury']['paired_active_mean_MAE_cm'][a]:.3f} |")
 lines+=['','C_B2/C_B3는 기존 기하를 그대로 사용하고 gain/bias만 FIT에 맞춘 RGB 진단 대조군이다. Feature same-trust의 깊이 평균은 기존 B3와 사실상 같아 이번 지원 사례에서 기하 탐색 범위 제한 자체가 결과 개선의 주된 차이로 보이지 않는다. 이는 RGB 목적함수·새 픽셀 sampling·밝기 nuisance가 함께 바뀐 비교이며 단일 원인의 인과 입증은 아니다.','',
 '| 방법 | KITTI 평균 법선오차(°) | Middlebury 평균 법선오차(°) |','|---|---:|---:|']
 for a in ARMS:lines.append(f"| {a} | {stats['kitti']['paired_active_mean_normal_error_deg'][a]:.3f} | {stats['middlebury']['paired_active_mean_normal_error_deg'][a]:.3f} |")
 lines+=['','KITTI P9c는 평균 깊이 오차가 감소했지만 평균 법선 오차는 개선하지 않았다. 이미지 정렬/깊이 개선을 평면 방향 정확도 개선과 동일하게 해석하지 않는다. Middlebury의 지원 사례는 Playtable/PlaytableP/Vintage의3개 ID로, 앞의 둘은 유사 장면이므로 독립3장면 일반화로 볼 수 없다.','',
 '## B2 fallback을 포함한 전체21개 결과','', '| 방법 | KITTI14개 평균 MAE(cm) | Middlebury7개 평균 MAE(cm) |','|---|---:|---:|']
 for a in ARMS:lines.append(f"| {a} | {stats['kitti']['all_retained_mean_MAE_cm'][a]:.3f} | {stats['middlebury']['all_retained_mean_MAE_cm'][a]:.3f} |")
 lines+=['','이 표에서 P9b/c의 미지원5개는 원래 B2를 유지했다. 따라서 전체 평균과 위의 paired 지원 평균을 혼동하지 않는다. 실패 시 GT로 B2/B3 중 더 좋은 것을 골라 유지하지 않았다.','',
 '## Paired 변화·보류 RGB 손실','', '| 데이터셋 | 비교 | 평균 MAE 차이(cm) | ID bootstrap 명목95% 구간 | 깊이 개선 | RGB 보류 개선 | 깊이·방향 개선+RGB 보류 비악화 |','|---|---|---:|---|---:|---:|---:|']
 for ds,s in stats.items():
  for name,p in s['paired_changes'].items():
   lo,hi=p['nominal_ID_bootstrap_95_cm'];n=p['cases'];lines.append(f"| {ds} | {name} | {p['mean_delta_MAE_cm']:+.3f} | [{lo:+.3f},{hi:+.3f}] | {p['lower_depth_MAE_cases']}/{n} | {p['lower_heldout_photo_cost_cases']}/{n} | {p['joint_lower_depth_and_normal_no_larger_heldout_photo_cases']}/{n} |")
 lines+=['','방향 개선 판정은1e-6°보다 큰 감소만 센다. 방향 고정군의 부동소수점 반올림 차이를 방향 개선으로 세지 않는다. 평균은 장면 ROI MAE의 동일 가중 평균이며 단위가 다른 데이터셋을 합치지 않았다. ID bootstrap10000회 seed0은 기술 통계다. 이미 본 데이터·연속 KITTI ID·동일 물리 장면 변형이 포함돼 독립 일반화 신뢰구간이나 유의성 입증으로 해석하지 않는다.','',
 '## P9d: 밝기·무텍스처·가림 관련 진단','', '| 데이터셋 | 군 | 밝기 파라미터 한도 도달 | warp 한도 도달 | 보류 고잔차 비율 평균 | 보류 RGB MAE: 보정없음→gain/bias |','|---|---|---:|---:|---:|---|']
 for ds,s in stats.items():
  for a,d in s['P9d'].items():lines.append(f"| {ds} | {a} | {d['active_brightness_bounds_hit']} | {d['active_warp_trust_boundary_hit']} | {d['mean_heldout_high_residual_fraction']:.3f} | {d['mean_heldout_raw_no_affine_RGB_MAE']:.4f}→{d['mean_heldout_affine_RGB_MAE']:.4f} |")
 lines+=['','고잔차는 RGB residual norm>0.03의 비율이다. 실제 가림 GT가 없으므로 가림 검출 정확도라고 표현하지 않는다. source gradient, 밝기 자유도를 제거한 plane Jacobian singular ratio, gain/bias 한도, trust 한도를 보존했다. 원래 pixel support가 충분한16개에서 low-texture gate는 통과했으나 이는 깊이 정확도 보장이 아니다.','',
 '| 미지원 사례 | FIT/보류 픽셀 | FIT/보류 tile | 이유 |','|---|---:|---:|---|']
 for r in rows:
  if r['support']['passed']:continue
  ss=r['support'];lines.append(f"| {r['dataset']}/{r['sample']} | {ss['counts']['fit']}/{ss['counts']['heldout']} | {ss['tiles']['fit']}/{ss['tiles']['heldout']} | INSUFFICIENT_SPATIAL_PHOTO_SUPPORT; B2 retained |")
 lines+=['','작은 평면 ROI에서 공간적으로 분리된 학습/보류 관측을 만들 수 없는 사례는 영역을 늘리거나 검증 기준을 낮춰 통과시키지 않았다. 이전8개 미지원 사례(ETH3D 두 장면 포함)는 현재 비교 범위 밖에 그대로 남겼다.','',
 '## 사례별 악화와 시각 산출물','']
 for ds in ['kitti','middlebury']:
  active=[r for r in rows if r['dataset']==ds and r['support']['passed']]
  for a,b in [('P9B_PHOTO_FIXED','C_B2'),('P9C_PHOTO_FREE','C_B3')]:
   bad=[r['sample'] for r in active if r['metrics'][a]['ROI_MAE_m']>r['metrics'][b]['ROI_MAE_m']];lines.append(f"- {ds} {a}가 {b}보다 악화: {', '.join(bad) if bad else '없음'}.")
 lines+=['','- `RGB_warp_alignment.png`: source RGB, target warp,8px checkerboard, 공통 RGB error 범위. Oracle GT plane 열은 사후 진단 전용이다.',
 '- `depth_GT_errors.png`: 동일 ROI/카메라/scale의 깊이와 GT 오차. GT 결측은 회색이다.',
 '- `ROI_pointmaps.npz`: 각 군의 source0 유한 ROI pointmap과 동일 RGB·pixel_uv·frame_id. 전체 장면이나 원래 VGGT 예측 pose pointmap과 구분한다.',
 '- 각 사례 `metrics.json`, `oracle_GT_plane_warp.json`, support/solver log, source snapshot·원본/잔차·GT 출처, `per_case_metrics.csv`, `aggregate.json`.',
 '', '## 기술 검증과 결론','',
 '합성 정답 평면에서 RGB warp의 해석미분을 수치미분과 비교했고, 자유/방향고정 광도 보정으로 정답 warp를 복원했다. 무텍스처 Jacobian 퇴화와 합성 수정 패치의 고잔차 반응도 확인했다. 실제 사례는 원본 캐시·잔차복원·비대상 불변·양의깊이·corner/dense warp trust·평가 전후 fit hash 불변을 검사했다. P9b scalar 관측 가능성은 `preGT_observability_check.json`에서 GT 평가 전에 별도로 검증했다.',
 '', 'RGB 직접 정렬은 기존 고정 대응점과 다른 관측을 사용해 KITTI에서 깊이 개선을 보였지만, 법선 정확도와 실내 성능의 일관된 개선은 확인하지 못했다. 광도 일치가 곧 기하 정답이라는 가정은 성립하지 않는 사례가 있다. 현재는 연구 방향의 가능성과 한계를 함께 보인 탐색 결과이며 새 독립 장면·노출 변화·실제 가림 GT·다중시점/장기 SLAM 검증은 남는다. 이번에도 기존 평면 내부를 대상으로 했고 RGB 경계 기반 재분할이나 물체 경계 복원은 실험하지 않았다.',
 '', f'Canonical fit: `{fit.name}`. 평가·그림·보고서: `{out.name}`.']
 report.write_text('\n'.join(lines)+'\n')
 with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write(f'\n- P9a–P9d RGB 직접 정렬 완료 — [보고서](P9/{out.name}/report.md), photo보정16/21, 기존8미지원 유지; 독립검증 아님.\n')
 print(json.dumps({'report':str(report),'datasets':list(stats)},indent=2))
if __name__=='__main__':main()
