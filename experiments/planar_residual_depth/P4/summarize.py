"""Produce P4 report, figures, CSV and bounded exploratory correlations from frozen evaluation."""
import sys,json,csv,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 out=BASE/sys.argv[1];rows=json.loads((out/'case_results.json').read_text());aggregate=json.loads((out/'aggregate.json').read_text());statuses=json.loads((out/'all_attempt_statuses.json').read_text());batch=BASE/json.loads((out/'config_resolved.json').read_text())['parent_batch'];shutil.copy2(__file__,out/'source/summarize.py');report=ROOT/'docs/experiments/planar_residual_depth/P4'/out.name/'report.md';report.parent.mkdir(parents=True,exist_ok=True)
 supports={r['dataset']+r['sample']:r.get('support',{}) for r in statuses}
 for s in ['000005','000006','000007']:supports['kitti'+s]=json.loads((BASE/f'kitti-{s}-P3obs-20261117-001/evaluation/support.json').read_text())
 flat=[];groups={}
 for r in rows:
  sup=supports[r['dataset']+r['sample']];mm={m['arm']:m for m in r['metrics']};entry={'dataset':r['dataset'],'sample':r['sample'],'mask_provenance':r['mask_provenance'],'FIT':sup['counts']['fit'],'heldout':sup['counts']['heldout'],'hull_fraction':sup['FIT_hull_area_fraction'],'parallax_deg':sup['median_FIT_parallax_deg'],'GT_valid_pixels':r['reference']['GT_valid_ROI_pixels'],'GT_plane_RMSE_cm':100*r['reference']['GT_plane_RMSE_m'],'GT_normal_bootstrap_p95_deg':r['reference']['GT_normal_change_p95_deg'],'H2_verdict':r['decision']['H2_verdict']}
  for a,m in mm.items():
   for k in ['MAE_m','RMSE_m','AbsRel','normal_error_deg','heldout_mean_px','heldout_median_px','thickness_95_m','own_plane_RMSE_m','reference_center_depth_offset_m','prediction_to_GT_plane_mean_abs_m']:entry[a+'_'+k]=m[k]
  entry['B3_minus_B2_MAE_cm']=100*(mm['B3']['MAE_m']-mm['B2']['MAE_m']);entry['B3_minus_B2_normal_deg']=mm['B3']['normal_error_deg']-mm['B2']['normal_error_deg'];flat.append(entry)
 with (out/'per_case.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 for ds in ['kitti','middlebury']:
  rr=[r for r in flat if r['dataset']==ds];groups[ds]={}
  for feature in ['FIT','hull_fraction','parallax_deg','B0_normal_error_deg']:
   st=spearmanr([r[feature] for r in rr],[r['B3_minus_B2_MAE_cm'] for r in rr]);groups[ds][feature]={'Spearman_rho':float(st.statistic),'nominal_p':float(st.pvalue),'status':'exploratory, small selected sample, uncorrected multiple comparisons; not causal'}
 dump(out/'exploratory_correlations.json',groups)
 for ds in ['kitti','middlebury']:
  rr=[r for r in flat if r['dataset']==ds];names=[r['sample'] for r in rr];x=np.arange(len(rr));fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
  for a in ['B0','B1','B2','B3']:axs[0,0].plot(x,[100*r[a+'_MAE_m'] for r in rr],'.-',label=a)
  axs[0,0].set_yscale('log');axs[0,0].set_ylabel('ROI MAE cm (log)');axs[0,0].legend();axs[0,0].set_xticks(x,names,rotation=60)
  delta=np.array([r['B3_minus_B2_MAE_cm'] for r in rr]);axs[0,1].bar(x,delta,color=np.where(delta<0,'tab:blue','tab:red'));axs[0,1].axhline(0,color='black',lw=.7);axs[0,1].set_ylabel('B3 - B2 MAE cm; negative favors B3');axs[0,1].set_xticks(x,names,rotation=60)
  nd=[r['B3_minus_B2_normal_deg'] for r in rr];axs[1,0].scatter(nd,delta)
  for r,n,d in zip(rr,nd,delta):axs[1,0].annotate(r['sample'],(n,d),fontsize=7)
  axs[1,0].axhline(0,color='gray',lw=.7);axs[1,0].axvline(0,color='gray',lw=.7);axs[1,0].set_xlabel('B3 - B2 normal error deg');axs[1,0].set_ylabel('B3 - B2 MAE cm')
  for a in ['B0','B1','B2','B3']:axs[1,1].plot(x,[100*r[a+'_thickness_95_m'] for r in rr],'.-',label=a)
  axs[1,1].set_ylabel('Plane thickness 95% cm');axs[1,1].set_xticks(x,names,rotation=60);axs[1,1].legend();fig.suptitle(ds+': conditional on observation gate; fixed settings, one plane/case');fig.savefig(out/(ds+'_summary.png'),dpi=150);plt.close(fig)
 lines=['# P4 최종 분석: 잔차 깊이 평면 보정','', '**P3의 고정 평가 29개 사례를 모두 처리하고 P4 분석을 완료했다. 정확도 평가가 가능한 사례는21개이며, 미지원8개는 실패/미지원 목록에 보존했다. 전체29개에서 보정이 성공했다는 뜻이 아니다.**','',
 '원본 VGGT-Ω 깊이를 보존하고 제공된 카메라 보정·RGB 대응으로 잔차만 계산했다. B0는 원본×RGB FIT 기반 공통scale, B1은 raw에 적합한 평면 투영, B2는 그 방향을 고정한 거리 보정, B3는 방향·거리 공동 보정이다. GT는 보정 완료 후 평가 프로세스에서만 읽었고 GT에 따라 결과·초기값·설정·영역을 선택하지 않았다.','',
 '## 실행과 변경 이력','',
 '- P0 합성 기하·미분 검증과 P2 개발 solver를 유지했다. 기존 개발 보조 실험 P2a/P2b와 이전 P3 결과를 덮어쓰지 않았다.',
 '- KITTI 000005–000007은 기존 사용자 승인 영역·실행을 재사용했다. 사용자가 이후 영역 검토를 생략하도록 명시했으므로 나머지26개는 에이전트가 RGB만 보고 영역을 지정했다. 이를 사용자 승인·SAM 출력으로 표시하지 않는다.',
 '- 모든 새 영역은 새 batch의 보정·raw depth 읽기 전에 동결했다. 장면마다 최대3개라는 계획 한도 내에서 보수적인 유한 평면1개를 선택했다. 곡면·가림 때문에 명확한 영역을 잡기 어려운 Pipes는 평면 없음으로 남겼다.',
 '- LK·SIFT·FIT/보류 분리·시차·면적·support gate·scale 정책·Huber·solver 설정을 유지했다. 카메라를 source 카메라 좌표계로 표현한 것은 동일 제공 자세의 좌표 변환이며 추가 자세 최적화가 아니다. ETH3D는 이 좌표계에서 수행했으나 모두 관측 gate에서 중단했다.',
 '- 새 batch는26개 중18개 fit 완료,7개 지원 부족,1개 평면 없음이다. 기존3개를 포함해 KITTI14/15, Middlebury7/12, ETH3D0/2에서 GT 정확도를 평가했다. 지원 기준을 낮추거나 다른 장면으로 대체하지 않았다.',
 '- KITTI GT는 정확한 calibrated rectified camera-Z 역산과 최근접 대응을 사용했다. Middlebury는 공식 PFM endian 부호만 사용하며 scale 절댓값으로 시차를 곱하지 않았다. baseline을 m로 변환했다. ETH3D 원본 distorted GT는 undistorted RGB에 직접 대입하지 않았으며 이번에는 보정 지원 부족으로 GT 평가하지 않았다.',
 '', '## 데이터셋별 주 결과','', '| 데이터셋 | GT 평가/계획 | B0 평균 MAE(cm) | B1 | B2 | B3 | B3 MAE가 더 작은 사례(B2 대비) |','|---|---:|---:|---:|---:|---:|---:|']
 for ds,a in aggregate.items():
  if not a['GT_evaluated']:lines.append(f"| {ds} | 0/{a['predefined_cases']} | — | — | — | — | — |")
  else:
   means=a['mean_case_MAE_m'];lines.append(f"| {ds} | {a['GT_evaluated']}/{a['predefined_cases']} | {means['B0']*100:.3f} | {means['B1']*100:.3f} | {means['B2']*100:.3f} | {means['B3']*100:.3f} | {a['B3_lower_MAE_cases']}/{a['GT_evaluated']} |")
 lines+=['','평균은 유효 픽셀을 모두 합친 평균이 아니라 평가 가능한 각 장면 ROI MAE의 동일 가중 평균이다. 데이터셋을 합친 단일 정확도는 만들지 않았다. 지원 부족 사례는 이 정확도 평균에 포함되지 않으므로 조건부 성능이며 전체 입력에 대한 성공률과 구분한다.','']
 for ds in ['kitti','middlebury']:
  a=aggregate[ds];lo,hi=np.array(a['B3_minus_B2_nominal_ID_bootstrap_95_m'])*100;lines.append(f"- {ds}: B3−B2 평균 MAE 변화 {a['B3_minus_B2_mean_MAE_m']*100:+.3f}cm, ID 단위 bootstrap 명목95% 구간 [{lo:+.3f}, {hi:+.3f}]cm. 음수는 B3 개선. 중앙값 MAE는 B2 {a['median_case_MAE_m']['B2']*100:.3f}cm, B3 {a['median_case_MAE_m']['B3']*100:.3f}cm.")
 lines+=['','이 구간은 장면 ID를10000회 재표집한 기술 통계다. KITTI의 연속 ID, Middlebury의 Playtable/PlaytableP 등은 동일 공간/유사 관측일 수 있다. 독립 장면 신뢰구간·유의성 입증으로 해석하지 않는다. 개발셋 K01은 집계에서 제외했다.','',
 '## 가설별 결론','',
 f"**H1 평면화 — 정확도와 평면성의 분리가 관측됨.** B1에서 두께가 감소하면서 MAE가 증가한 사례는 KITTI {aggregate['kitti']['H1_cases']}/14, Middlebury {aggregate['middlebury']['H1_cases']}/7이다. 강제 평면화의 두께≈0은 GT 정확도 증거가 아니다. B1은 B0에 적합한 방향을 그대로 쓰므로 방향 오류가 B1에서 새로 커진다고 해석하지 않는다. MAE 변화에는 작은 적합 오차·거리·비평면성이 함께 관여할 수 있다.", '',
 f"**H2 방향 보정 — 조건부 이득, 보편적 우월성은 미확립.** B3가 B2보다 깊이·법선 오차를 모두 줄이고 보류 평균 재투영 오차도 악화시키지 않은 사례는 KITTI {aggregate['kitti']['H2_nominal_joint_cases']}/14, Middlebury {aggregate['middlebury']['H2_nominal_joint_cases']}/7이다. 해당 사례들은 GT 참조 방향의 공간 block bootstrap에서도 방향 개선이 분리되었다. KITTI 평균 차이는 거의0이고 명목구간이0을 포함하므로 `INCONCLUSIVE`; Middlebury에서는 현재 평가 가능한 작은 평면 집합에서 B3 이득이 관측된다. 모두 사용자 미검토 영역이므로 별도의 독립 검증은 남는다.", '',
 '**H3 관측 지원 — 현 파이프라인의 적용 범위 제한이 관측됨.** 29개 중8개가 평면/관측 지원 단계에서 보정되지 못했다. ETH3D 두 장면의 selected5-view/기존 LK/지정 영역 조합은0 FIT/보류였다. 이것은 카메라만으로 방향 추정이 불가능하다는 증거가 아니라 이 고정 파이프라인이 해당 입력에서 지원을 만들지 못했다는 결과다. 지원 수가 많다고 정확하다는 보장도 없다. 최소 지원을 통과한 사례만 있으므로 gate 효과나 매칭 오류의 인과관계를 여기서 확정할 수 없다. `exploratory_correlations.json`은 작은 선택 표본의 비보정 탐색 통계다.', '',
 '**앞선 세 도로 사례만으로 “B2가 항상 낫다”고 보는 해석은 수정해야 한다.** 넓혀 보면 KITTI에서 B3가 더 좋은 사례가 더 많지만, 일부 악화 사례의 크기 때문에 평균이 비슷해진다. 따라서 GT로 더 좋은 군을 사후 선택한 oracle 평균을 실제 방법 성능으로 제시하지 않는다.','',
 '## GT 참조 방향·평면성의 불확실성','',
 'GT 참조 평면은 같은 ROI의 유효GT에 robust 적합했다. 처리 격자16×16블록을100회 복원추출하고 ref 법선 변동과 B3−B2 법선 오차 차이의 percentile 구간을 저장했다. 각 블록 내 픽셀 상관을 유지했지만 더 큰 공간 상관·센서 체계 오차·마스크 오분할은 해결하지 않는다. 이는 ground truth의 완전한 불확실성 모델이 아니다. 방향 개선 구간이0을 포함하면 참조에 따른 방향 개선은 미분리로 기록했다. GT 평면 잔차·지지 hull·bootstrap tile 수는 각 사례 JSON에 있다. 두께≈0인 보정면의 시각적 매끈함을 정확도 개선으로 간주하지 않았다.','',
 '| 데이터셋/샘플 | B0 MAE(cm) | B1 | B2 | B3 | 법선 B2→B3(°) | GT평면 RMS(cm) | GT법선 변동p95(°) | H2 |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
 for r in flat:lines.append(f"| {r['dataset']}/{r['sample']} | {100*r['B0_MAE_m']:.3f} | {100*r['B1_MAE_m']:.3f} | {100*r['B2_MAE_m']:.3f} | {100*r['B3_MAE_m']:.3f} | {r['B2_normal_error_deg']:.3f}→{r['B3_normal_error_deg']:.3f} | {r['GT_plane_RMSE_cm']:.3f} | {r['GT_normal_bootstrap_p95_deg']:.3f} | {'joint' if r['H2_verdict']=='JOINT_IMPROVEMENT_REFERENCE_RESOLVED' else '未成立'} |")
 lines+=['','## 악화·미지원 사례','']
 for ds in ['kitti','middlebury']:
  rr=[r for r in flat if r['dataset']==ds]
  for a in ['B2','B3']:
   bad=[r['sample'] for r in rr if r[a+'_MAE_m']>r['B0_MAE_m']];lines.append(f"- {ds} {a}가 B0보다 악화: {', '.join(bad) if bad else '없음'}.")
 lines+=['','| 데이터셋/샘플 | 상태 | FIT/보류 | hull | 시차(°) |','|---|---|---:|---:|---:|']
 for r in statuses:
  if r['status']=='FIT_COMPLETED_GT_NOT_READ':continue
  s=r.get('support',{});c=s.get('counts',{});par=s.get('median_FIT_parallax_deg');lines.append(f"| {r['dataset']}/{r['sample']} | {r['status']} | {c.get('fit','—')}/{c.get('heldout','—')} | {s.get('FIT_hull_area_fraction',0):.4f} | {par if par is not None else '—'} |")
 lines+=['','평면이 없어 제외한 Middlebury Pipes와3D/곡면이 지배적인 장면을 억지로 평면화하지 않았다. `INSUFFICIENT_SUPPORT`는 현재 후보·관측 추출·split 기준에서 미지원이라는 뜻이며, 다른 영역이나 모델이 실패한다는 뜻은 아니다. 이번 고정 평가 안에서는 결과를 본 뒤 재지정하지 않았다.','',
 '## 정성 산출물과 기술 검증','',
 '- 사례별 `depth_GT_errors.png`: RGB, B0–B3 깊이, GT, 동일 범위 signed 오차. 회색은 ROI 밖 또는 GT 결측/평가 제외다.',
 '- `residual_cross_section.png`: 잔차맵과 ROI 평균 y의 고정 행 단면. 동일 깊이 축·범위를 유지했다.',
 '- `GT_arrays.npz`, `GT_reference.json`, `metrics.json`, `decision.json`: 평가 mask·방향·위치·평면성·원시 깊이 오차·보류 대응·불확실성.',
 '- fit의 `arms/*/depths.npz`: 원본·scale·잔차·보정 깊이·유한mask·공유 평면 계수. pointmap은 제공 광선으로 이 값에서 파생되며 직접 최적화하지 않았다.',
 '- frozen solver SHA256, 해석 Jacobian 수치대조·rank·양의깊이·잔차 복원·원본/비대상 불변을 검사했다. 평가 전후 모든 fit NPZ hash가 동일하다. P2 합성 정답평면·gradient·방향고정 검사도 다시 통과했다.',
 '', '## 해석 범위와 다음 연구 방향','',
 '이번 결과는 제공 카메라 보정, 작은 RGB 지정 평면, 공통 RGB 삼각측량 scale을 사용하는 국소 depth 보정 실험이다. VGGT 예측 자세까지 포함한 전체 모델 성능, 경계 복원, 자동 평면 분할, 전체 장면 pointmap 정확도, 장기 SLAM·loop closure를 검증하지 않는다. Middlebury의 mm 수준 오차도 이런 통제 조건과 작은 선택 ROI에서의 수치다. 샘플 ID 간 물리 공간 독립성은 확인되지 않았다.',
 '', '방향 자유도를 항상 풀거나 항상 고정하기보다, 새로운 개발 실험에서 관측의 방향 불확실성·공간 분포를 이용한 보수적 방향 갱신을 검토할 근거가 생겼다. 다만 이번 GT를 보고 군을 선택한 방법으로 성능을 주장할 수 없으므로 새로운 규칙·독립 평가 분할을 먼저 동결해야 한다. ETH3D 적용 범위를 늘리려면 selected views의 실제 겹침과 큰 회전/시점 변화 대응부터 별도 한 요인 실험으로 다뤄야 한다.','',
 f'실행: `results/planar_residual_depth/{batch.name}/`, 평가·집계: `results/planar_residual_depth/{out.name}/`.']
 report.write_text('\n'.join(lines)+'\n')
 with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write(f'\n- P3 전 고정 평가29개 처리·P4 최종 분석 완료(21 GT평가/8미지원) — [최종 보고서](P4/{out.name}/report.md), fit batch `{batch.name}`.\n')
 print(report)
if __name__=='__main__':main()
