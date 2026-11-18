"""Aggregate all P5-P8 outcomes without selecting rules by GT; render actual ROI comparisons."""
import sys,json,csv,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

def summarize(rows):
 result={}
 comparisons=[('P6_NEAR','B2'),('P6_CONF','B2'),('P6_NEAR_CONF','B2'),('P7_NEAR','P6_NEAR'),('P7_NEAR_CONF','P6_NEAR_CONF'),('P8_LIMITED_RAW','B2'),('P8_LIMITED_RAW','B3')]
 for ds in ['kitti','middlebury']:
  rr=[r for r in rows if r['dataset']==ds];arms=list(rr[0]['metrics']);d={'cases':len(rr),'mean_case_MAE_cm':{},'mean_normal_error_deg':{},'mean_far_MAE_cm':{},'refined_cases':{},'paired':{},'P5':{}}
  for a in arms:
   mm=[r['metrics'][a] for r in rr];d['mean_case_MAE_cm'][a]=float(np.mean([m['MAE_m'] for m in mm])*100);d['mean_normal_error_deg'][a]=float(np.mean([m['normal_error_deg'] for m in mm]));d['mean_far_MAE_cm'][a]=float(np.mean([m['FAR_MAE_m'] for m in mm])*100);d['refined_cases'][a]=sum(m['status']=='REFINED' for m in mm)
  for a,b in comparisons:
   delta=np.array([100*(r['metrics'][a]['MAE_m']-r['metrics'][b]['MAE_m']) for r in rr]);rng=np.random.default_rng(0);boot=delta[rng.integers(0,len(rr),(10000,len(rr)))].mean(1);d['paired'][a+'_vs_'+b]={'mean_delta_MAE_cm':float(delta.mean()),'nominal_ID_bootstrap_95_cm':np.percentile(boot,[2.5,97.5]).tolist(),'lower_MAE_cases':int(np.sum(delta<0)),'joint_lower_depth_and_normal_no_larger_heldout_mean_cases':sum(r['metrics'][a]['MAE_m']<r['metrics'][b]['MAE_m'] and r['metrics'][a]['normal_error_deg']<r['metrics'][b]['normal_error_deg'] and r['metrics'][a]['heldout_mean_px']<=r['metrics'][b]['heldout_mean_px'] for r in rr)}
  for a,b in [('NEAR','FAR'),('HIGH_CONF','LOW_CONF'),('NEAR_HIGH','ALL')]:
   pp=[r for r in rr if all('normal_error_global_GT_deg' in r['P5'][g] for g in [a,b])];d['P5'][a+'_vs_'+b]={'cases':len(pp),'lower_normal_error_cases':sum(r['P5'][a]['normal_error_global_GT_deg']<r['P5'][b]['normal_error_global_GT_deg'] for r in pp),'mean_normal_error_deg':{g:float(np.mean([r['P5'][g]['normal_error_global_GT_deg'] for r in pp])) for g in [a,b]}}
  d['P5_controls']=[]
  for family in [['NEAR','MID','FAR'],['LOW_CONF','MID_CONF','HIGH_CONF'],['ALL','NEAR','HIGH_CONF','NEAR_HIGH']]:
   for mode in ['COUNT_MATCHED','SPATIAL_CELL_MATCHED']:
    cc=[next(c for c in r['P5_controls'] if c['family']==family and c['mode']==mode) for r in rr];valid=[c for c in cc if c.get('mean_normal_error_deg') is not None];d['P5_controls'].append({'family':family,'mode':mode,'supported_cases':len(valid),'total_cases':len(rr),'mean_normal_error_deg':{g:float(np.mean([c['mean_normal_error_deg'][g] for c in valid])) for g in family} if valid else None})
  result[ds]=d
 return result

def visualize(row,out):
 case=ROOT/row['fit_dir'];info=json.loads((case/'case.json').read_text());v=info['views'][0];data=np.load(case/'inputs.npz');mask=data['masks'][0];parent=ROOT/info['parent_fit'];oldar=parent/'arms' if (parent/'arms').exists() else parent/'artifacts/arms';old_eval=ROOT/info['parent_P4_evaluation'];gt=np.load(old_eval/'GT_arrays.npz')['GT_depth'];valid=mask&np.isfinite(gt)&(gt>0);arms={}
 for a in ['B0','B2','B3']:arms[a]=np.load(oldar/a/'depths.npz')['corrected_depth'][0]
 for a in ['P6_NEAR','P7_NEAR','P8_LIMITED_RAW']:arms[a]=np.load(case/'arms'/(a+'.npz'))['corrected_depth'][0]
 dest=out/'cases'/(row['dataset']+'-'+row['sample']);rgb=np.array(Image.open(ROOT/v['processed']).convert('RGB'));ys,xs=np.where(mask);sl=np.s_[max(0,ys.min()-8):min(mask.shape[0],ys.max()+9),max(0,xs.min()-8):min(mask.shape[1],xs.max()+9)]
 vals=np.concatenate([gt[valid]]+[d[mask] for d in arms.values()]);lo,hi=np.percentile(vals,[1,99]);bound=max(1e-6,float(np.percentile(np.concatenate([abs(d[valid]-gt[valid]) for d in arms.values()]),95)));cm=plt.get_cmap('viridis').copy();cm.set_bad('#999999');ec=plt.get_cmap('RdBu_r').copy();ec.set_bad('#999999');fig,axs=plt.subplots(2,7,figsize=(18,6),layout='constrained');axs[0,0].imshow(rgb[sl]);axs[0,0].set_title('RGB crop');axs[1,0].imshow(np.where(valid,gt,np.nan)[sl],cmap=cm,vmin=lo,vmax=hi);axs[1,0].set_title('GT; gray=not evaluated')
 for j,(a,d) in enumerate(arms.items(),1):
  im=axs[0,j].imshow(np.where(mask,d,np.nan)[sl],cmap=cm,vmin=lo,vmax=hi);axs[0,j].set_title(a.replace('P8_LIMITED_RAW','P8 limit 1deg'));ie=axs[1,j].imshow(np.where(valid,d-gt,np.nan)[sl],cmap=ec,vmin=-bound,vmax=bound);m=row['metrics'][a];axs[1,j].set_title(f"MAE {m['MAE_m']*100:.2f} cm\nnormal {m['normal_error_deg']:.2f} deg")
 for ax in axs.ravel():ax.axis('off')
 fig.colorbar(im,ax=axs[0,1:],label='camera-Z m',shrink=.65);fig.colorbar(ie,ax=axs[1,1:],label='prediction - GT m',shrink=.65);fig.suptitle(row['dataset']+' '+row['sample']+' | same frozen ROI/cameras/scale, post-hoc exploratory');fig.savefig(dest/'P6_P8_depth_GT.png',dpi=140);plt.close(fig)
 # P5 selection maps are image-backed, not synthetic examples.
 indices=np.load(case/'selected_pixel_indices.npz');selection=np.full(mask.shape,np.nan)
 for k,g in enumerate(['NEAR','MID','FAR']):selection.ravel()[indices['view0_'+g]]=k
 fig,axs=plt.subplots(1,4,figsize=(14,4),layout='constrained');axs[0].imshow(rgb);axs[0].set_title('Frozen RGB plane');axs[1].imshow(selection,cmap=plt.get_cmap('coolwarm',3),vmin=0,vmax=2);axs[1].set_title('Estimated-Z thirds: blue near / red far');im=axs[2].imshow(np.where(mask,data['depth_conf'][0],np.nan),cmap=cm);axs[2].set_title('VGGT depth_conf score');fig.colorbar(im,ax=axs[2],shrink=.6);sel=np.zeros(mask.shape,bool);sel.ravel()[indices['view0_NEAR_HIGH']]=True;axs[3].imshow(rgb);axs[3].imshow(np.where(sel,1,np.nan),cmap='autumn',alpha=.65,vmin=0,vmax=1);axs[3].set_title('Near AND high confidence')
 for ax in axs:ax.axis('off')
 fig.suptitle(row['dataset']+' '+row['sample']+' | selection uses raw prediction only');fig.savefig(dest/'P5_groups.png',dpi=130);plt.close(fig)
 # Finite source ROI pointmaps retain raw RGB and pixel provenance; no whole-scene claim.
 h,w=mask.shape;y,x=np.mgrid[:h,:w];uv=np.c_[x[mask],y[mask]];q=rays(uv,v);X=np.array([np.array(v['C'])+d[mask,None]*q for d in arms.values()]);np.savez_compressed(dest/'ROI_pointmaps.npz',arm_names=np.array(list(arms)),X=X,RGB=rgb[mask],pixel_uv=uv,frame_id=np.array(v['source']),scope=np.array('source0 frozen ROI, provided reference camera coordinates'))

def main():
 out=BASE/sys.argv[1];rows=json.loads((out/'summary.json').read_text());fit=BASE/json.loads((out/'config_resolved.json').read_text())['parent_fit'];stats=summarize(rows);dump(out/'aggregate.json',stats);shutil.copy2(__file__,out/'source/report.py')
 flat=[]
 for r in rows:
  for a,m in r['metrics'].items():flat.append({'dataset':r['dataset'],'sample':r['sample'],'arm':a,**{k:v for k,v in m.items() if k!='arm'}})
 fields=sorted(set(k for r in flat for k in r))
 with (out/'per_case_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
 for r in rows:visualize(r,out)
 for ds,st in stats.items():
  rr=[r for r in rows if r['dataset']==ds];fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained');aa=['B2','B3','P6_NEAR','P7_NEAR','P8_LIMITED_RAW'];axs[0].bar(np.arange(len(aa)),[st['mean_case_MAE_cm'][a] for a in aa]);axs[0].set_xticks(np.arange(len(aa)),aa,rotation=35);axs[0].set_ylabel('mean case ROI MAE cm');axs[1].bar(np.arange(len(aa)),[st['mean_far_MAE_cm'][a] for a in aa]);axs[1].set_xticks(np.arange(len(aa)),aa,rotation=35);axs[1].set_ylabel('estimated far third MAE cm')
  for a in ['P6_NEAR','P7_NEAR','P8_LIMITED_RAW']:
   dd=[100*(r['metrics'][a]['MAE_m']-r['metrics']['B2']['MAE_m']) for r in rr];axs[2].plot(np.arange(len(rr)),dd,'.-',label=a)
  axs[2].axhline(0,color='gray',lw=.7);axs[2].set_xticks(np.arange(len(rr)),[r['sample'] for r in rr],rotation=60);axs[2].set_ylabel('MAE delta vs B2 cm');axs[2].legend();fig.suptitle(ds+' | same already-seen cases; no independent validation');fig.savefig(out/(ds+'_P6_P8_summary.png'),dpi=140);plt.close(fig)
 report=ROOT/'docs/experiments/planar_residual_depth/P5_P8'/out.name/'report.md';report.parent.mkdir(parents=True,exist_ok=True)
 lines=['# P5a–P8: 거리·신뢰도와 보수적 평면 방향 보정','', '**P5a/P5b/P5c 진단과 P6/P7/P8 보정을21개 기존 사례에서 완료했다. 이전8개 미지원 사례는 그대로 보존했다. 새 독립 장면 검증이 아닌, P4 결과를 이미 본 뒤 규칙을 사전에 고정한 후속 탐색 실험이다.**','',
 '## 사전 고정 조건','',
 '사용자가 P8까지 연속 실행을 요청했다. 새 GPU 추론·학습·마스크 변경 없이 기존 제공 K/pose·동일 영역·FIT/보류·RGB 기반 공통scale을 유지했다. VGGT-Ω 원본 depth와 depth_conf를 보존했다. confidence는 픽셀 깊이 점수이며 카메라 자세의 신뢰도나 보정된 확률이 아니다. 거리 구분은 GT가 아닌 scaled raw camera-Z의 순위 삼등분이다.',
 '', 'P6–P8의 모든 규칙은 P5의 GT 진단 전에 `protocol.json`으로 동결했다. P6는 near / high-confidence / 교집합의 세 고정 군이며 주 비교는 near다. P7는 같은 선택 규칙의 두 시점 법선을 raw bootstrap 안정성·공간분포·상대 confidence로 가중 통합하고 거리만 보정한다. 주 비교는 P7_NEAR 대 P6_NEAR다. P8는 전체 raw 평면 방향에서 최대1°만 움직이게 하고 FIT 재투영 손실을 최소화한다. 기존 B2/B3는 고정/자유 대조군이다. GT로 군이나 각도 한도를 선택하지 않았다.',
 '', '## P5a: 가까운 점의 방향이 더 좋은가?','', '| 데이터셋 | 가까운 점 법선오차 평균(°) | 먼 점 | 가까운 쪽이 나은 사례 | 점수/공간 통제 |','|---|---:|---:|---:|---|']
 for ds,s in stats.items():
  a=s['P5']['NEAR_vs_FAR'];lines.append(f"| {ds} | {a['mean_normal_error_deg']['NEAR']:.3f} | {a['mean_normal_error_deg']['FAR']:.3f} | {a['lower_normal_error_cases']}/{a['cases']} | 동일 개수 재표집20회; 아래 공간 한계 참조 |")
 lines+=['','이번 고정 ROI에서 raw 추정 가까운 구간만 쓰는 방향은 전체 점보다 일관되게 낫지 않았다. 그러나 near/mid/far는 영상의 서로 다른 위치에 놓이므로 거리 효과를 인과적으로 분리하지 못한다. 4×4 공통 cell에서 세 군의 개수와 분포를 동시에 맞추는 통제는 KITTI0/14, Middlebury0/7로 불가능했다. 이를 `NOT_IDENTIFIABLE_SPATIAL_CONFOUNDING`으로 남겼으며 “근거리일수록 본질적으로 덜 정확하다”로 일반화하지 않는다. GT도 local subset으로 평면을 따로 적합한 참고 각도를 기록했다.','',
 '## P5b: 높은 depth_conf의 방향이 더 좋은가?','', '| 데이터셋 | 높은 confidence 법선오차(°) | 낮은 confidence | 높은 쪽이 나은 사례 |','|---|---:|---:|---:|']
 for ds,s in stats.items():
  a=s['P5']['HIGH_CONF_vs_LOW_CONF'];lines.append(f"| {ds} | {a['mean_normal_error_deg']['HIGH_CONF']:.3f} | {a['mean_normal_error_deg']['LOW_CONF']:.3f} | {a['lower_normal_error_cases']}/{a['cases']} |")
 lines+=['','점 개수만 맞춘 비교와 공간 cell별 개수까지 맞춘 비교를 분리했다. 높은 깊이 confidence가 올바른 평면 방향의 신뢰도를 의미한다고 볼 수 없다. 이 점수가 모든 깊이/매칭 과제에서 무용하다는 결론도 아니다.','',
 '## P5c 및 공간 통제 상세','', '| 데이터셋 | 비교군 | 통제 | 지원 사례 | 군별 평균 법선오차(°) |','|---|---|---|---:|---|']
 for ds,s in stats.items():
  for c in s['P5_controls']:
   vals=', '.join(f'{g}={v:.3f}' for g,v in c['mean_normal_error_deg'].items()) if c['mean_normal_error_deg'] else '미분리';lines.append(f"| {ds} | {' / '.join(c['family'])} | {c['mode']} | {c['supported_cases']}/{c['total_cases']} | {vals} |")
 lines+=['','각 표의 지원 사례 집합과 공통 cell 영역은 다를 수 있으므로 표 사이 숫자를 그대로 상대 성능으로 비교하지 않는다. count control은 각 군 최대1024점·20회, spatial control은 공통 cell마다 동일 개수·20회다. 표는 raw normal 지원 기준을 모든 군에서 통과한 paired 반복에 대한 장면별 평균의 평균이다. 같은 cell 안의 정확한 위치까지 같지는 않아 잔여 공간 혼재가 있다. 교집합 선택도 일관된 방향 개선을 보장하지 않았다.','',
 '## P6–P8 깊이 보정 결과','', '| 방법 | KITTI 평균 MAE(cm),14사례 | Middlebury 평균 MAE(cm),7사례 |','|---|---:|---:|']
 for a in ['B0','B1','B2','B3','P6_NEAR','P6_CONF','P6_NEAR_CONF','P7_NEAR','P7_NEAR_CONF','P8_LIMITED_RAW']:lines.append(f"| {a} | {stats['kitti']['mean_case_MAE_cm'][a]:.3f} | {stats['middlebury']['mean_case_MAE_cm'][a]:.3f} |")
 lines+=['','B0는 원본×고정 공통scale, B1은 기존 raw 평면 투영이다. P6/P7은 고른/통합한 방향을 고정하고 기존 FIT 대응으로 거리만 보정한다. 각 arm의 원본/비대상 픽셀·카메라·split은 동일하다.','',
 '### P6 먼 영역 전파','', '| 데이터셋 | B2 먼 구간 MAE(cm) | P6_NEAR | P6_CONF | P6_NEAR_CONF |','|---|---:|---:|---:|---:|']
 for ds,s in stats.items():lines.append(f"| {ds} | {s['mean_far_MAE_cm']['B2']:.3f} | {s['mean_far_MAE_cm']['P6_NEAR']:.3f} | {s['mean_far_MAE_cm']['P6_CONF']:.3f} | {s['mean_far_MAE_cm']['P6_NEAR_CONF']:.3f} |")
 lines+=['','가까운 점을 선택하면 먼 구간까지 더 정확해진다는 가설은 현재 규칙으로 일관되게 지지되지 않았다. 원인을 거리 자체로 단정하지 않으며, 좁아진 지지 영역·raw 편향·법선 추정 민감도가 함께 남는다.','',
 '### P7 두 시점 법선 통합','',
 '현재 지원 사례는 stereo 두 장이다. 5-view ETH3D는 이전부터 지원 부족이므로 이번에도 미검증이다. 각 시점의 raw 부분 평면을 제공 자세로 같은 좌표계에 표현하고, 16×16 공간 block bootstrap30회의 법선 변동·영상 hull·ROI 내 confidence percentile로 정한 가중치로 축 평균했다. 이는 raw 추정의 안정성 가중이며 실제 정확도의 보정된 확률이 아니다. P7는 KITTI에서 P6보다 평균 오차를 소폭 줄였지만 기존 B2를 넘지 못했다. Middlebury에서도 B3 수준에 도달하지 못했다. 카메라 자세 confidence를 사용하거나 최적화한 실험은 아니다.','',
 '### P8 방향 변화1° 제한','',
 '양의 깊이 선형 제약에 n0·p − cos(1°)||p|| ≥ 0의 해석미분 제약을 추가했다. B2/raw B1과 범위 내에 있을 때만 B3를 초기값으로 사용하고 FIT 손실만으로 선택했다. 이는 항상 방향을 고정하거나 완전히 자유롭게 두는 것 사이의 고정된 보수적 비교다. KITTI에서는 평균 개선, Middlebury에서는 자유 B3보다 큰 오차가 나타났다. 1°를 최적 한도로 주장하지 않으며 GT로 조절하지 않았다.','',
 '## Paired 변화와 불확실성','', '| 데이터셋 | 비교 | 평균 MAE 차이(cm) | ID bootstrap 명목95% 구간 | 개선 사례 |','|---|---|---:|---|---:|']
 for ds,s in stats.items():
  for comp,a in s['paired'].items():
   lo,hi=a['nominal_ID_bootstrap_95_cm'];lines.append(f"| {ds} | {comp} | {a['mean_delta_MAE_cm']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {a['lower_MAE_cases']}/{s['cases']} |")
 lines+=['','음수는 앞의 방법이 더 정확하다는 뜻이다. 기존 장면을 이미 분석했고, KITTI 연속ID·Middlebury 유사 장면 사이의 독립성이 보장되지 않아 이 구간을 독립 일반화 신뢰구간이나 유의성 입증으로 해석하지 않는다. 후속 규칙은 이미 관찰한 P4에서 아이디어를 얻었으므로 모든 결과가 탐색적이다.','',
 '## 정확도 악화와 기술 검증','']
 for ds,s in stats.items():
  rr=[r for r in rows if r['dataset']==ds]
  for a in ['P6_NEAR','P7_NEAR','P8_LIMITED_RAW']:
   bad=[r['sample'] for r in rr if r['metrics'][a]['MAE_m']>r['metrics']['B2']['MAE_m']];lines.append(f"- {ds} {a}가 B2보다 악화: {', '.join(bad) if bad else '없음'}.")
 lines+=['','21개 모두6개 새 보정군 계산 완료. 각 군의 상태와 fallback은 개별 metric/solver log에 보존했다. P8 법선 변화 한도·양의깊이·동일 공유평면 교차·잔차복원·비대상 불변·raw/conf 캐시 일치·평가 전후 fit hash 불변을 검증했다. 별도 합성 실험에서 cone 제약 gradient와 내부 정답 복원, 바깥 정답에 대한1° 경계 수렴을 확인했다.','',
 '## 실험 이력·산출물','',
 '- canonical fit: `'+fit.name+'`. 초기 `P5-P8fit-all-cases-20261117-001`은 경로를 얻기 위해 GT 지표가 포함된 P4 JSON container를 읽는 행정 I/O가 있었다. 알고리즘은 경로 필드만 사용했고 GT를 최적화하지 않았지만, GT 미로딩 경계를 엄격히 유지하려고 보존 후 canonical 실행에서는 pre-GT P3 manifest에서만 경로를 구성해 재실행했다.',
 '- `protocol.json`/config와 source snapshot: P5 GT 진단 전에 P6–P8 규칙 고정. 기존 결과와 별도 폴더, 원본 데이터/실행은 변경하지 않았다.',
 '- 각 사례: `P5_diagnostics.json`, `P5_controls.json`, `metrics.json`, `P5_groups.png`, `P6_P8_depth_GT.png`, `ROI_pointmaps.npz`.',
 '- fit 각 사례: `inputs.npz`(원본·depth_conf·기존scale·baseline·mask), 군별 잔차/보정depth·p, raw normal·선택 pixel IDs·가중치·solver log.',
 '- `aggregate.json`, `per_case_metrics.csv`, 데이터셋별 요약 그림. RGB/GT/오차는 동일 ROI·공통 표시 범위를 사용하고 GT 결측은 회색이다.',
 '', '## 결론','',
 '현재 데이터에서 “가깝고 신뢰도가 높으면 더 정확한 평면 방향”이라는 선택 규칙은 확인되지 않았다. 특히 거리와 영상 내 분포를 동시에 맞춘 비교가 불가능하므로 거리 자체의 효과는 `INCONCLUSIVE`다. 높은 depth_conf만으로 방향의 정확도를 보장할 수도 없다. 전체 점으로 얻은 방향을 중심으로 작은 범위의 관측 보정을 허용하는 P8은 KITTI에서 유망한 탐색 결과를 보였지만, 실내에서는 큰 방향 수정이 필요한 사례를 막는다. 독립 장면에서의 평가·카메라 신뢰도·다중시점 ETH3D·장기 SLAM·학습 모델은 이번 단계의 검증 범위 밖이다.']
 report.write_text('\n'.join(lines)+'\n');dump(out/'report_manifest.json',{'report':str(report.relative_to(ROOT)),'source':str((out/'source/report.py').relative_to(ROOT)),'stages':['P5a','P5b','P5c','P6','P7','P8'],'status':'REPORT_COMPLETED','independent_validation':False})
 with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write(f'\n- P5a–P8 후속 탐색 완료 — [보고서](P5_P8/{out.name}/report.md), fit `{fit.name}`; 기존21개 지원/8개 미지원 유지, 독립검증 아님.\n')
 print(json.dumps({'report':str(report),'aggregate':stats},indent=2))
if __name__=='__main__':main()
