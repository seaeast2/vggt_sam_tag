"""Post-fit GT evaluation; cannot modify optimizer outputs or select a new plane."""
import json,sys,time,hashlib,shutil
from pathlib import Path
import numpy as np,cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver import rays
from fit import plane
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 fitrun=Path(sys.argv[1]);fitrun=fitrun if fitrun.is_absolute() else ROOT/fitrun
 manifest=json.loads((fitrun/'manifest.json').read_text());assert manifest['status']=='FIT_COMPLETED_GT_NOT_READ'
 sourcehashes={str(p.relative_to(fitrun)):sha(p) for p in fitrun.rglob('*') if p.is_file()}
 inputs=json.loads((fitrun/'artifacts/inputs.json').read_text());views=inputs['views'];config=json.loads((fitrun/'config_resolved.json').read_text())
 arms={arm:np.load(fitrun/'artifacts/arms'/arm/'depths.npz') for arm in ['B0','B1','B2','B3']}
 shape=arms['B0']['baseline'][0].shape;h,w=shape;yy,xx=np.mgrid[:h,:w];uv=np.c_[xx.ravel(),yy.ravel()];v=views[0];K=np.array(v['K']);A=np.array(v['A']);Ko=np.linalg.inv(A)@K
 gtpath=ROOT/'data/planar_residual_depth/kitti2015/source/training/disp_occ_0/000000_10.png';disp=cv2.imread(str(gtpath),cv2.IMREAD_UNCHANGED).astype(float)/256
 B=np.linalg.norm(np.array(views[1]['C'])-np.array(v['C']));K1o=np.linalg.inv(np.array(views[1]['A']))@np.array(views[1]['K']);doffs=K1o[0,2]-Ko[0,2]
 valid=(disp>0)&np.isfinite(disp)&(disp+doffs>0);GT=np.full_like(disp,np.nan);GT[valid]=Ko[0,0]*B/(disp[valid]+doffs)
 coords=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(A).T;mx=coords[:,0].reshape(shape).astype(np.float32);my=coords[:,1].reshape(shape).astype(np.float32)
 gt=cv2.remap(GT.astype(np.float32),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))
 old=np.load(BASE/'P1b-baseline-direction-20261117-004/artifacts/K01/arrays.npz')['GT'];assert np.array_equal(gt,old,equal_nan=True)
 mask=arms['B0']['masks'][0];validgt=np.isfinite(gt)&(gt>0);evalmask=mask&validgt;assert evalmask.sum()==9078
 Xgt=np.array(v['C'])+gt[evalmask,None]*rays(uv[evalmask.ravel()],v);pg=plane(Xgt);ng=pg/np.linalg.norm(pg)
 rows=[];obs={x['arm']:x for x in json.loads((fitrun/'evaluation/observation_metrics.json').read_text())}
 for arm,a in arms.items():
  cor=a['corrected_depth'][0];baseline=a['baseline'][0];raw=a['raw_depth'];s=float(a['scale'])
  assert np.array_equal(cor[~mask],baseline[~mask])
  assert np.array_equal(a['baseline'],raw.astype(float)*s)
  p=a['p'];p=plane(np.array(v['C'])+cor[mask,None]*rays(uv[mask.ravel()],v)) if arm=='B0' else p
  normal=p/np.linalg.norm(p);angle=float(np.degrees(np.arccos(np.clip(abs(normal@ng),0,1))))
  err=cor[evalmask]-gt[evalmask];erfull=cor[validgt]-gt[validgt]
  plane_abs=abs(Xgt@normal+1/np.linalg.norm(p))
  rows.append({'arm':arm,'ROI_valid_GT_pixels':int(evalmask.sum()),'ROI_MAE_m':float(np.mean(abs(err))),
   'ROI_RMSE_m':float(np.sqrt(np.mean(err**2))),'ROI_AbsRel':float(np.mean(abs(err)/gt[evalmask])),'ROI_bias_m':float(np.mean(err)),
   'normal_error_deg':angle,'GT_point_to_model_plane_mean_m':float(np.mean(plane_abs)),
   'whole_frame_valid_GT_MAE_m':float(np.mean(abs(erfull))),'whole_frame_valid_GT_pixels':int(validgt.sum()),**obs[arm]})
 seq=1;prefix='kitti-000000-P2e-'+time.strftime('%Y%m%d')+'-'
 while (BASE/(prefix+f'{seq:03}')).exists():seq+=1
 rid=prefix+f'{seq:03}';out=BASE/rid;out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/evaluate.py')
 dump(out/'evaluation/metrics.json',rows);dump(out/'evaluation/GT_reference.json',{'p':pg.tolist(),'normal':ng.tolist(),'GT_sha256':sha(gtpath),'GT_plane_RMSE_m':float(np.sqrt(np.mean((Xgt@ng+1/np.linalg.norm(pg))**2))),
 'mask':'all finite positive GT in approved source polygon; descriptive development ROI, not an independent GT test split','GT_for_optimization':False,'right_view_GT':'not evaluated; disp_occ_1 is time1 left disparity, not right-camera depth'})
 np.savez_compressed(out/'evaluation/GT_arrays.npz',GT_depth=gt,GT_valid=validgt,ROI_mask=mask,evaluation_mask=evalmask)
 # Shared displays include all ROI depths; no per-arm range tuning.
 depthvals=np.concatenate([gt[evalmask]]+[a['corrected_depth'][0][mask] for a in arms.values()]);lo,hi=np.percentile(depthvals,[1,99])
 bound=float(np.percentile(np.concatenate([abs(a['corrected_depth'][0][evalmask]-gt[evalmask]) for a in arms.values()]),95));cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#999999');ecmap=plt.get_cmap('RdBu_r').copy();ecmap.set_bad('#999999')
 fig,axes=plt.subplots(2,5,figsize=(17,7),layout='constrained');rgb=np.array(Image.open(ROOT/views[0]['processed']))
 axes[0,0].imshow(rgb);axes[0,0].set_title('RGB: approved K01 road');poly=inputs['polygons'][0];axes[0,0].plot([p[0] for p in poly]+[poly[0][0]],[p[1] for p in poly]+[poly[0][1]],color='red')
 im=axes[1,0].imshow(np.where(evalmask,gt,np.nan),cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest');axes[1,0].set_title('GT (gray = not evaluated)')
 for col,(arm,a) in enumerate(arms.items(),1):
  depth=a['corrected_depth'][0];im=axes[0,col].imshow(np.where(mask,depth,np.nan),cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest');axes[0,col].set_title(arm+' depth')
  imerr=axes[1,col].imshow(np.where(evalmask,depth-gt,np.nan),cmap=ecmap,vmin=-bound,vmax=bound,interpolation='nearest');r=next(r for r in rows if r['arm']==arm);axes[1,col].set_title(f"{arm}: MAE {r['ROI_MAE_m']*100:.2f} cm\nnormal error {r['normal_error_deg']:.3f} deg")
 for ax in axes.ravel():ax.axis('off')
 fig.colorbar(im,ax=axes[0,1:],label='Camera-Z (m), shared range',shrink=.6,extend='both');fig.colorbar(imerr,ax=axes[1,1:],label='Prediction - GT (m): red far / blue near',shrink=.6,extend='both')
 fig.suptitle('K01 exploratory plane correction; fixed cameras, one FIT-only scale; 9078 GT pixels')
 fig.savefig(out/'K01_depth_GT_errors.png',dpi=150);plt.close(fig)
 # Save a source-view pointcloud per arm, original colors and exact pixel identity.
 for arm,a in arms.items():
  depth=a['corrected_depth'][0];X=np.array(v['C'])+depth.ravel()[:,None]*rays(uv,v);color=rgb.reshape(-1,3)
  folder=out/'artifacts'/arm;folder.mkdir(parents=True)
  with (folder/'pointmap.ply').open('w') as f:
   f.write('ply\nformat ascii 1.0\nelement vertex '+str(len(X))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int pixel_u\nproperty int pixel_v\nend_header\n')
   for xyz,col,xy in zip(X,color,uv):f.write(' '.join(map(str,xyz))+' '+' '.join(map(str,col))+' '+' '.join(map(str,xy))+'\n')
 unchanged=all(sha(fitrun/path)==h for path,h in sourcehashes.items());assert unchanged
 dump(out/'manifest.json',{'run_id':rid,'stage':'P2e','parent_fit':fitrun.name,'fit_outputs_hashes':sourcehashes,'fit_outputs_unchanged':unchanged,'GT_used_for_optimization':False,'scope':'one approved development plane, left-camera GT only','status':'EVALUATION_COMPLETED'})
 report=ROOT/'docs/experiments/planar_residual_depth/P2'/rid/'report.md';report.parent.mkdir(parents=True,exist_ok=True)
 r0,r1,r2,r3=rows;lines=['# K01 평면 깊이 보정 B0–B3 결과','', '사용자가 source와 target 다각형을 동일 평면으로 확인한 뒤 실행했다. 학습 모델 없이 CPU에서 원본 VGGT 깊이를 보존하고 잔차만 저장했다. FIT 관측으로 후보를 생성·선택한 실행이 완료된 후 별도 프로세스에서 GT를 읽었다.','',
 f"FIT {config['FIT_count']}개·보류 {config['heldout_count']}개, FIT hull {config['hull_area_fraction']:.3f}, 시차 {config['median_FIT_parallax_deg']:.3f}°. 공통 scale {config['scale']:.12f}; 새 LK 보류의 양 시점3px 이내 scale-FIT 관측은 제외하도록 검사했고 실제 제외 {config['scale_excluded_near_holdout']}개다. scale에는 남은 FIT SIFT {config['scale_FIT_tracks']}개를 사용했다.",'',
 '| 군 | ROI MAE(cm) | RMSE(cm) | 법선 오차(°) | 보류 재투영 median(px) | 보류 mean(px) |','|---|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['arm']} | {100*r['ROI_MAE_m']:.4f} | {100*r['ROI_RMSE_m']:.4f} | {r['normal_error_deg']:.4f} | {r['heldout_median_px']:.4f} | {r['heldout_mean_px']:.4f} |")
 lines+=['','B0는 원본 깊이×공통scale, B1은 source raw 깊이에 적합한 공유 평면, B2는 B1 방향을 고정한 거리 보정, B3는 방향·거리 공동 보정이다. B1–B3는 같은 공유 world 평면을 양쪽 승인 마스크에 적용했다. 원본·비대상 픽셀·RGB·카메라는 유지했다.', '',
 f"B3의 B0 대비 ROI MAE 변화: {(r0['ROI_MAE_m']-r3['ROI_MAE_m'])/r0['ROI_MAE_m']*100:.3f}% 감소. B3의 B2 대비 변화: {(r2['ROI_MAE_m']-r3['ROI_MAE_m'])/r2['ROI_MAE_m']*100:.3f}% 감소. 음수는 악화다.", '',
 '## 해석 범위','', '현재 ROI는 이미 관찰한 개발 장면이며 새로운 평가셋·전체 KITTI·SLAM 성능으로 일반화하지 않는다. GT 수치는 승인 source ROI의 유효GT 전체 9078점에서 계산한 기술 통계다. 픽셀은 독립 표본이 아니고 GT heldout 실험으로 부르지 않는다. 보류 재투영은21개 LK track의 관측 일치이며 GT 정확도와 분리한다. LK 패치·ROI 내부의 레일·노면 굴곡·초기 카메라·매칭 오류는 남는다. M01/P01/P02는 지원 부족으로 보정하지 않았다.', '',
 'GT는 KITTI training disp_occ_0의 기준시각 왼쪽 카메라다. disp_occ_1을 오른쪽 카메라 GT로 오인하지 않는다. 제공 K와 baseline, 원본→처리 격자 변환으로 camera-Z를 계산했다. 오른쪽 depthmap은 생성하지만 독립 GT 평가는 하지 않았다.', '',
 '합성 해석미분·목적함수gradient·정답평면 복원·방향고정 검사를 통과한 후 실데이터를 실행했다. 실데이터의 Jacobian 수치 대조·rank·양의깊이·잔차복원·비대상 불변을 확인했다. B3 두 초기값은 FIT만으로 선택했고 보류/GT로 방향을 재선택하지 않았다. 평가 전후 FIT 출력 hash 불변을 검사했다.', '',
 f'FIT 실행: `results/planar_residual_depth/{fitrun.name}/`. 평가·그림·source-view PLY: `results/planar_residual_depth/{rid}/`.']
 report.write_text('\n'.join(lines)+'\n')
 with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write(f'\n- P2 K01 B0–B3 실행·별도GT평가 완료 — [보고서](P2/{rid}/report.md), FIT `{fitrun.name}`, 평가 `{rid}`.\n')
 print(json.dumps({'run_id':rid,'metrics':rows,'report':str(report)},indent=2))
if __name__=='__main__':main()
