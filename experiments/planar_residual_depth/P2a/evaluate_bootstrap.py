"""Evaluate completed P2a bootstrap outputs using frozen GT. Never reruns solver."""
import json,hashlib,time,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summary(values):
 x=np.array(values);return {'n':len(x),'median':float(np.median(x)),'mean':float(np.mean(x)),'std':float(np.std(x,ddof=1)),
  'min':float(np.min(x)),'max':float(np.max(x)),'p2_5':float(np.percentile(x,2.5)),'p97_5':float(np.percentile(x,97.5))}
def main():
 fitrun=Path(sys.argv[1]);fitrun=fitrun if fitrun.is_absolute() else ROOT/fitrun
 manifest=json.loads((fitrun/'manifest.json').read_text());assert manifest['status']=='FIT_COMPLETED_GT_NOT_READ'
 frozen={str(p.relative_to(fitrun)):sha(p) for p in fitrun.rglob('*') if p.is_file()}
 parent=BASE/manifest['parent'];gtparent=BASE/'P2e-kitti-000000-20261117-001'
 gtfile=gtparent/'evaluation/GT_arrays.npz';g=np.load(gtfile);gt=g['GT_depth'];valid=g['evaluation_mask'];mask=g['ROI_mask']
 reference=json.loads((gtparent/'evaluation/GT_reference.json').read_text());ng=np.array(reference['normal'])
 original={r['arm']:r for r in json.loads((gtparent/'evaluation/metrics.json').read_text())}
 basearr=np.load(parent/'artifacts/arms/B1/depths.npz');rawbase=basearr['baseline'];masks=basearr['masks']
 configs=json.loads((fitrun/'config_resolved.json').read_text());allfits=json.loads((fitrun/'fit_summary.json').read_text());inputs=json.loads((parent/'artifacts/inputs.json').read_text())
 fitids={t['track_id'] for t in inputs['tracks'] if t['split']=='fit'};heldids={t['track_id'] for t in inputs['tracks'] if t['split']=='heldout'}
 rows=[];non_target_checks=0
 for rep in allfits:
  assert set(rep['draw_track_ids'])<=fitids;assert not set(rep['draw_track_ids'])&heldids
  for arm,r in rep.get('arms',{}).items():
   if not r['success']:continue
   file=fitrun/'artifacts'/f"replicate{rep['replicate']:03}"/arm/'depths.npz';a=np.load(file);cor=a['corrected_depth'];delta=a['residual_depth'];p=a['p']
   for j in [0,1]:
    assert np.array_equal(cor[j][~masks[j]],rawbase[j][~masks[j]])
    assert np.allclose(rawbase[j]+delta[j],cor[j],rtol=0,atol=1e-12);assert (cor[j][masks[j]]>0).all();non_target_checks+=1
   error=cor[0][valid]-gt[valid];normal=p/np.linalg.norm(p);angle=float(np.degrees(np.arccos(np.clip(abs(normal@ng),0,1))))
   rows.append({'replicate':rep['replicate'],'arm':arm,'ROI_MAE_m':float(np.mean(abs(error))),
    'ROI_RMSE_m':float(np.sqrt(np.mean(error**2))),'ROI_bias_m':float(np.mean(error)),
    'normal_error_deg':angle,'heldout_median_px':r['heldout_median_px'],'heldout_mean_px':r['heldout_mean_px'],
    'unique_FIT':rep['unique_FIT'],'FIT_hull_fraction':rep['FIT_hull_fraction']})
 aggregate={arm:{metric:summary([r[metric] for r in rows if r['arm']==arm]) for metric in ['ROI_MAE_m','ROI_RMSE_m','normal_error_deg','heldout_median_px','heldout_mean_px']} for arm in ['B2','B3']}
 paired=[]
 for rep in allfits:
  rs={r['arm']:r for r in rows if r['replicate']==rep['replicate']}
  if set(rs)=={'B2','B3'}:paired.append({'replicate':rep['replicate'],'B3_minus_B2_MAE_m':rs['B3']['ROI_MAE_m']-rs['B2']['ROI_MAE_m'],
   'B3_minus_B2_normal_deg':rs['B3']['normal_error_deg']-rs['B2']['normal_error_deg']})
 counts={'paired':len(paired),'B3_MAE_better_than_B2':sum(r['B3_minus_B2_MAE_m']<0 for r in paired),
 'B3_normal_better_than_B2':sum(r['B3_minus_B2_normal_deg']<0 for r in paired),'B3_MAE_better_than_B0':sum(r['ROI_MAE_m']<original['B0']['ROI_MAE_m'] for r in rows if r['arm']=='B3')}
 prefix='kitti-000000-P2ae-'+time.strftime('%Y%m%d')+'-';seq=1
 while (BASE/(prefix+f'{seq:03}')).exists():seq+=1
 rid=prefix+f'{seq:03}';out=BASE/rid;out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/evaluate_bootstrap.py')
 dump(out/'evaluation/replicate_metrics.json',rows);dump(out/'evaluation/summary.json',{'arms':aggregate,'paired':counts,'paired_differences':paired,'quantile_interpretation':'empirical conditional FIT-track bootstrap spread; NOT scene-generalization confidence interval'})
 fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 for k,(metric,mult,label) in enumerate([('ROI_MAE_m',100,'Depth MAE (cm)'),('normal_error_deg',1,'Normal error (deg)'),('heldout_mean_px',1,'Heldout mean reprojection (px)')]):
  for arm,color in [('B2','tab:blue'),('B3','tab:orange')]:
   rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['replicate']);ax[k].plot([r['replicate'] for r in rr],[r[metric]*mult for r in rr],'.-',label=arm,color=color,alpha=.8)
   ax[k].axhline(original[arm][metric]*mult,color=color,ls='--',alpha=.6)
  ax[k].set(xlabel='Bootstrap replicate',ylabel=label);ax[k].grid(alpha=.2);ax[k].legend()
 fig.suptitle('Same K01 plane, fixed scale/cameras/masks/heldout; only FIT track weights resampled\nDashed = original full-FIT P2 result; 30 conditional bootstrap draws')
 fig.savefig(out/'bootstrap_stability.png',dpi=150);plt.close(fig)
 # Visuals were picked in the FIT stage before any GT evaluation.
 reps=json.loads((fitrun/'visual_representatives.json').read_text())['replicates']
 rgb=np.array(Image.open(ROOT/inputs['views'][0]['processed']));fig,axes=plt.subplots(2,4,figsize=(15,6),layout='constrained')
 ecmap=plt.get_cmap('RdBu_r').copy();ecmap.set_bad('#999999')
 axes[0,0].imshow(rgb);axes[0,0].set_title('Source RGB');axes[1,0].imshow(np.where(valid,gt,np.nan),cmap='viridis');axes[1,0].set_title('GT depth: same evaluated ROI')
 for col,rep in enumerate(reps,1):
  for row,arm in enumerate(['B2','B3']):
   a=np.load(fitrun/'artifacts'/f'replicate{rep:03}'/arm/'depths.npz');err=a['corrected_depth'][0]-gt
   im=axes[row,col].imshow(np.where(valid,err,np.nan),cmap=ecmap,vmin=-1,vmax=1,interpolation='nearest')
   r=next(r for r in rows if r['replicate']==rep and r['arm']==arm)
   axes[row,col].set_title(f"rep {rep} / {arm}: {100*r['ROI_MAE_m']:.2f} cm\nnormal {r['normal_error_deg']:.2f} deg")
 for a in axes.ravel():a.axis('off')
 fig.colorbar(im,ax=axes[:,1:].ravel().tolist(),label='Prediction - GT (m): red too far, blue too near',shrink=.7,extend='both')
 fig.suptitle('Representatives chosen by B3 FIT cost before GT; gray = outside ROI / missing GT')
 fig.savefig(out/'bootstrap_representative_errors.png',dpi=150);plt.close(fig)
 unchanged=all(sha(fitrun/p)==h for p,h in frozen.items());assert unchanged
 dump(out/'verification.json',{'FIT_only_draw_ids':True,'heldout_never_drawn':True,'paired_resampling_same_for_B2_B3':True,'non_target_residual_positive_depth_checks':non_target_checks,
  'fit_outputs_hashes_unchanged':unchanged,'GT_read_after_all_fits':True,'unique_FIT_min':min(r['unique_FIT'] for r in allfits),'unique_FIT_max':max(r['unique_FIT'] for r in allfits)})
 dump(out/'manifest.json',{'run_id':rid,'stage':'P2ae','status':'EVALUATION_COMPLETED','parent':fitrun.name,'parent_hashes':frozen,'GT_reference_sha256':sha(gtfile),'GT_used_for_optimization':False})
 report=ROOT/'docs/experiments/planar_residual_depth/P2a'/rid/'report.md';report.parent.mkdir(parents=True,exist_ok=True)
 lines=['# P2a FIT 대응점 재표본화 안정성 결과','', '원본 P2의 K01 양쪽 마스크·카메라·baseline·scale·B1 초기평면·solver·보류21개를 고정했다. FIT47개 track에서 복원추출로47번 뽑는 paired bootstrap을30회 수행하고 같은 draw를 B2/B3에 사용했다. 새 장면이나 새 대응을 추가하지 않았다. 반복 결과를 GT로 선택하지 않았다.','',
 f"수치 성공: {manifest['success_count']}/30 paired 반복. 실제 고유 FIT track 수 범위는 {min(r['unique_FIT'] for r in allfits)}–{max(r['unique_FIT'] for r in allfits)}개다. GT 평가는 모든 fit 출력이 저장된 뒤 별도 프로세스로 수행했다. 동일 source-view 유효GT9078점을 사용했다.", '',
 '| 군 | MAE 중앙값(cm) | MAE 경험적 2.5–97.5%(cm) | 법선오차 중앙값(°) | 법선 경험적 2.5–97.5%(°) |','|---|---:|---|---:|---|']
 for arm in ['B2','B3']:
  r=aggregate[arm];m=r['ROI_MAE_m'];n=r['normal_error_deg'];lines.append(f"| {arm} | {100*m['median']:.3f} | {100*m['p2_5']:.3f}–{100*m['p97_5']:.3f} | {n['median']:.3f} | {n['p2_5']:.3f}–{n['p97_5']:.3f} |")
 lines+=['', f"B3가 B2보다 MAE가 낮은 반복: {counts['B3_MAE_better_than_B2']}/{counts['paired']}. B3 법선이 더 정확한 반복: {counts['B3_normal_better_than_B2']}/{counts['paired']}. B3가 원본 B0보다 MAE가 낮은 반복: {counts['B3_MAE_better_than_B0']}/{counts['paired']}.", '',
 '## 해석 제한','', '경험적 범위는 고정된47개 FIT 관측의 가중치를 바꾼 조건부 변동이다. 독립적인 장면 일반화 신뢰구간이 아니다. 가까운 LK 패치들은 상관될 수 있으므로 독립 관측 가정을 만족한다고 주장하지 않는다. 잘못된 매칭·카메라 오차·GT 노면의 비평면성 같은 공통 편향은 재표본화로 제거되지 않는다. bootstrap은 원인 식별 실험이 아니며 변동만으로 특정 오차 원인을 확정하지 않는다.', '',
 'B2는 모든 반복에서 원본 방향을 고정했으므로 법선 오차의 변동이 없는 것은 설계상 당연하다. B3의 방향 및 깊이 변동을 해석할 때 이 점을 구분한다. GT는 기존 승인ROI의 전체 유효점 통계이며 새로운 독립 테스트셋 평가가 아니다.', '',
 '시각 대표 반복은 GT 평가 전 B3의 전체 원래FIT Huber 비용 최소·중앙·최대 순위로 고정했다. GT가 좋은/나쁜 반복을 골라 보여주지 않았다. 실패나 gate 미달은 재추출하지 않고 원래 draw를 보존하도록 했으며 이번에는30회 모두 통과했다.', '',
 f'FIT run: `results/planar_residual_depth/{fitrun.name}/`; 평가·그림: `results/planar_residual_depth/{rid}/`.']
 report.write_text('\n'.join(lines)+'\n')
 with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write(f'\n- P2a 대응점 재표본화30회 완료 — [보고서](P2a/{rid}/report.md), FIT `{fitrun.name}`, 평가 `{rid}`.\n')
 print(json.dumps({'run_id':rid,'summary':aggregate,'paired':counts,'report':str(report)},indent=2))
if __name__=='__main__':main()
