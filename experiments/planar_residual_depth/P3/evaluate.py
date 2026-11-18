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
 case=json.loads((BASE/manifest['parent']/'case.json').read_text());sample=case['sample'];candidate=case['candidate']
 approval=json.loads((BASE/manifest['parent']/'approval.json').read_text())
 sourcehashes={str(p.relative_to(fitrun)):sha(p) for p in fitrun.rglob('*') if p.is_file()}
 inputs=json.loads((fitrun/'artifacts/inputs.json').read_text());views=inputs['views'];config=json.loads((fitrun/'config_resolved.json').read_text())
 arms={arm:np.load(fitrun/'artifacts/arms'/arm/'depths.npz') for arm in ['B0','B1','B2','B3']}
 shape=arms['B0']['baseline'][0].shape;h,w=shape;yy,xx=np.mgrid[:h,:w];uv=np.c_[xx.ravel(),yy.ravel()];v=views[0];K=np.array(v['K']);A=np.array(v['A']);Ko=np.linalg.inv(A)@K
 gtpath=ROOT/f'data/planar_residual_depth/kitti2015/source/training/disp_occ_0/{sample}_10.png';disp=cv2.imread(str(gtpath),cv2.IMREAD_UNCHANGED).astype(float)/256
 # Exact rectified camera-Z inversion, retaining calibrated horizontal and Z offsets.
 K1o=np.linalg.inv(np.array(views[1]['A']))@np.array(views[1]['K'])
 assert np.allclose(K1o,Ko) and np.allclose(v['R'],np.eye(3)) and np.allclose(views[1]['R'],np.eye(3))
 dc=np.array(views[1]['C'])-np.array(v['C']);oy,ox=np.mgrid[:disp.shape[0],:disp.shape[1]];rx=(ox-Ko[0,2])/Ko[0,0]
 valid=(disp>0)&np.isfinite(disp);GT=np.full_like(disp,np.nan)
 GT[valid]=Ko[0,0]*(dc[0]-rx[valid]*dc[2])/disp[valid]+dc[2]
 coords=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(A).T;mx=coords[:,0].reshape(shape).astype(np.float32);my=coords[:,1].reshape(shape).astype(np.float32)
 gt=cv2.remap(GT.astype(np.float32),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))
 mask=arms['B0']['masks'][0];validgt=np.isfinite(gt)&(gt>0);evalmask=mask&validgt;assert evalmask.sum()>=3
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
 seq=1;prefix=f'kitti-{sample}-P3e-'+time.strftime('%Y%m%d')+'-'
 while (BASE/(prefix+f'{seq:03}')).exists():seq+=1
 rid=prefix+f'{seq:03}';out=BASE/rid;out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/evaluate.py')
 dump(out/'evaluation/metrics.json',rows);dump(out/'evaluation/GT_reference.json',{'p':pg.tolist(),'normal':ng.tolist(),'GT_sha256':sha(gtpath),'GT_plane_RMSE_m':float(np.sqrt(np.mean((Xgt@ng+1/np.linalg.norm(pg))**2))),
 'mask':'all finite positive GT in approved source polygon; predefined evaluation sample; same mask all arms, not certified independent physical scene','GT_for_optimization':False,'depth_conversion':'exact rectified inversion Z=fx*(Bx-rx*Bz)/disparity+Bz; no depth interpolation or hole filling','right_view_GT':'not evaluated; disp_occ_1 is time1 left disparity, not right-camera depth'})
 np.savez_compressed(out/'evaluation/GT_arrays.npz',GT_depth=gt,GT_valid=validgt,ROI_mask=mask,evaluation_mask=evalmask)
 # Shared displays include all ROI depths; no per-arm range tuning.
 depthvals=np.concatenate([gt[evalmask]]+[a['corrected_depth'][0][mask] for a in arms.values()]);lo,hi=np.percentile(depthvals,[1,99])
 bound=float(np.percentile(np.concatenate([abs(a['corrected_depth'][0][evalmask]-gt[evalmask]) for a in arms.values()]),95));cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#999999');ecmap=plt.get_cmap('RdBu_r').copy();ecmap.set_bad('#999999')
 fig,axes=plt.subplots(2,5,figsize=(17,7),layout='constrained');rgb=np.array(Image.open(ROOT/views[0]['processed']))
 axes[0,0].imshow(rgb);axes[0,0].set_title(f'RGB: approved {candidate} road');poly=inputs['polygons'][0];axes[0,0].plot([p[0] for p in poly]+[poly[0][0]],[p[1] for p in poly]+[poly[0][1]],color='red')
 im=axes[1,0].imshow(np.where(evalmask,gt,np.nan),cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest');axes[1,0].set_title('GT (gray = not evaluated)')
 for col,(arm,a) in enumerate(arms.items(),1):
  depth=a['corrected_depth'][0];im=axes[0,col].imshow(np.where(mask,depth,np.nan),cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest');axes[0,col].set_title(arm+' depth')
  imerr=axes[1,col].imshow(np.where(evalmask,depth-gt,np.nan),cmap=ecmap,vmin=-bound,vmax=bound,interpolation='nearest');r=next(r for r in rows if r['arm']==arm);axes[1,col].set_title(f"{arm}: MAE {r['ROI_MAE_m']*100:.2f} cm\nnormal error {r['normal_error_deg']:.3f} deg")
 for ax in axes.ravel():ax.axis('off')
 fig.colorbar(im,ax=axes[0,1:],label='Camera-Z (m), shared range',shrink=.6,extend='both');fig.colorbar(imerr,ax=axes[1,1:],label='Prediction - GT (m): red far / blue near',shrink=.6,extend='both')
 fig.suptitle(f'{candidate}: fixed evaluation ID{sample}; fixed cameras and FIT-only scale; {evalmask.sum()} GT pixels')
 fig.savefig(out/f'{candidate}_depth_GT_errors.png',dpi=150);plt.close(fig)
 # Save a source-view pointcloud per arm, original colors and exact pixel identity.
 for arm,a in arms.items():
  depth=a['corrected_depth'][0];X=np.array(v['C'])+depth.ravel()[:,None]*rays(uv,v);color=rgb.reshape(-1,3)
  folder=out/'artifacts'/arm;folder.mkdir(parents=True)
  with (folder/'pointmap.ply').open('w') as f:
   f.write('ply\nformat ascii 1.0\nelement vertex '+str(len(X))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int pixel_u\nproperty int pixel_v\nend_header\n')
   for xyz,col,xy in zip(X,color,uv):f.write(' '.join(map(str,xyz))+' '+' '.join(map(str,col))+' '+' '.join(map(str,xy))+'\n')
 unchanged=all(sha(fitrun/path)==h for path,h in sourcehashes.items());assert unchanged
 dump(out/'manifest.json',{'run_id':rid,'stage':'P3e','parent_fit':fitrun.name,'fit_outputs_hashes':sourcehashes,'fit_outputs_unchanged':unchanged,'GT_used_for_optimization':False,'scope':f'predefined evaluation sample ID{sample}, left-camera GT only','status':'EVALUATION_COMPLETED'})
 report=ROOT/'docs/experiments/planar_residual_depth/P3'/rid/'report.md';report.parent.mkdir(parents=True,exist_ok=True)
 lines=[f'# P3: KITTI {sample} 도로 평면 고정 설정 평가','',
 f'사전에 정한 평가 ID {sample}를 순서대로 선택했다. RGB에서 지정한 {candidate} 좌·우 도로 다각형에 대해 사용자가 승인했다(원문: {approval["user_statement"]}). GT와 raw depth를 보지 않고 영역을 지정했다. P2의 매칭·지원 기준·스케일 정책·평면 적합·보정 solver를 유지했고 이 장면의 결과로 재조정하지 않았다.','',
 f"FIT {config['FIT_count']}개 / 보류 {config['heldout_count']}개. FIT hull 비율 {config['hull_area_fraction']:.3f}, 중앙 시차 {config['median_FIT_parallax_deg']:.3f}°. FIT SIFT {config['scale_FIT_tracks']}개로 공통 scale {config['scale']:.12f}. 보류 3px 이내 scale-FIT 제외 {config['scale_excluded_near_holdout']}개.",'',
 '| 군 | ROI MAE(cm) | RMSE(cm) | 법선 오차(°) | 보류 재투영 중앙(px) | 보류 평균(px) |','|---|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['arm']} | {100*r['ROI_MAE_m']:.4f} | {100*r['ROI_RMSE_m']:.4f} | {r['normal_error_deg']:.4f} | {r['heldout_median_px']:.4f} | {r['heldout_mean_px']:.4f} |")
 lines+=['','B0: 원본×공통scale. B1: raw source로 적합한 평면 투영. B2: B1 방향 고정, 거리만 보정. B3: 방향·거리 공동 보정. 모든 군의 마스크·카메라·scale과 GT 평가 픽셀을 동일하게 유지했다. raw와 잔차를 별도 저장했으며 비대상 픽셀은 변경하지 않았다.','',
 f"공통 평가 픽셀 {evalmask.sum()}개. GT는 KITTI disp_occ_0의 왼쪽 기준시각 disparity다. calib의 Z방향 baseline까지 포함한 정확한 rectified 역산 Z=fx*(Bx-rx*Bz)/d+Bz를 적용했다. 원본 GT를 최근접으로 처리 격자에 대응시키며 보간·결측 채움은 하지 않았다. 이전 P2e의 baseline norm 근사를 여기서는 사용하지 않는다. 이 변환 변경은 최적화에 영향을 주지 않는다. 오른쪽 GT 정확도는 평가하지 않았다.",'',
 'GT는 보정 완료 후 별도 프로세스에서만 읽었다. FIT 출력의 평가 전후 hash 불변, 양의 깊이·Jacobian 수치 대조·rank·잔차 복원·비대상 불변 검사를 통과했다. solver.py는 P2 실행 스냅샷과 SHA256이 동일하다. 보류/GT로 평면을 재선택하지 않았다.','',
 '평가 ID로의 확장 사례 1개이며, 새로운 독립 공간이나 전체 KITTI/SLAM 정확도 검증은 아니다. 개발 ID와 촬영 구간·실제 공간이 독립인지 검증하지 않았다. 영역은 수동 승인 평면이고 픽셀·LK 패치 사이에는 상관이 있다. 제공 카메라 보정을 사용하므로 RGB만으로 카메라 자세까지 추정하는 실험과 구별한다.','',
 f'FIT: `results/planar_residual_depth/{fitrun.name}/`. 평가·그림·PLY: `results/planar_residual_depth/{rid}/`.']
 report.write_text('\n'.join(lines)+'\n')
 with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write(f'\n- P3 KITTI {sample} 고정 설정 평가 완료 — [보고서](P3/{rid}/report.md), FIT `{fitrun.name}`, 평가 `{rid}`.\n')
 print(json.dumps({'run_id':rid,'metrics':rows,'report':str(report)},indent=2))
if __name__=='__main__':main()
