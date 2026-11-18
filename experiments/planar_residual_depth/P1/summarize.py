"""Report actual P1 preparation, including incomplete gates and preserved failures."""
import json,hashlib
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[3];base=ROOT/'results/planar_residual_depth'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parents=['P1-middlebury-all-20261117-001','P1-kitti2015-first20-20261117-001','P1-eth3d-office-20261117-002','P1-eth3d-pipes-20261117-003','P1-eth3d-delivery_area-20261117-001','P1-eth3d-courtyard-20261117-001']
raws=['P1-middlebury-raw-20261117-001','P1-kitti2015-first20-raw-20261117-001','P1-kitti2015-first20-raw-20261117-002','P1-eth3d-four-scenes-raw-20261117-001']
out=base/'P1-audit-20261117-001';out.mkdir(exist_ok=False)
records=[];seen=set();frames=0;peaks=[]
for rid in raws:
 manifest=json.loads((base/rid/'manifest.json').read_text())
 for sample in manifest['samples']:
  key=('middlebury' if rid.startswith('middlebury') else 'kitti2015' if rid.startswith('kitti') else 'eth3d',sample['sample'])
  assert key not in seen;seen.add(key)
  p=base/rid/'artifacts/raw'/sample['sample']/'predictions.npz';assert sha(p)==sample['output_sha256']
  with np.load(p) as a:
   assert np.isfinite(a['depth']).all() and (a['depth']>0).all()
   count=a['depth'].shape[1];frames+=count
  peaks.append(sample['peak_allocated_bytes']);records.append({'dataset':key[0],'sample':key[1],'run':rid,'frames':count,'sha256':sha(p)})
assert len(records)==39 and frames==90
candidates=json.loads((base/'review/P1-candidates-v002/candidates.json').read_text())
for c in candidates:
 run=base/c['source_run'];poly=np.array(c['polygon'],np.float32)
 if c['id']=='M01':
  ts=json.loads((run/'artifacts/inputs/Adirondack/tracks.json').read_text());pts=[(t['split'],t['uv0']) for t in ts]
 else:
  sample='000000' if c['id']=='K01' else 'pipes';ts=json.loads((run/'artifacts/observations/tracks.json').read_text())[sample]
  pts=[(t['split'],o['uv']) for t in ts for o in t['observations'] if o['view']==0]
 c['support']={s:sum(cv2.pointPolygonTest(poly,tuple(uv),False)>=0 for sp,uv in pts if sp==s) for s in ['fit','heldout']}
 c['support_status']='INSUFFICIENT_SUPPORT' if c['support']['fit']<12 or c['support']['heldout']<6 else 'COUNT_GATE_PASSED_OTHER_GATES_PENDING'
 summary={'status':'P1_PARTIAL_AWAITING_MASK_REVIEW_AND_GT_GRID_VALIDATION','raw_groups':len(records),'raw_frames':frames,
 'peak_allocated_bytes_max':max(peaks),'raw_records':records,'canonical_input_runs':parents,'candidate_support':candidates,
 'pending':['human planar masks in multiple views','plane-level spatial coverage and parallax checks','FIT-only scale freeze','GT depth grid adapter validation'],
 'GT_used_for_optimization':False,'correction_executed':False,
 'preserved_issues':['ETH3D pipes first path join failed','ETH3D original calibration runs superseded by minus 0.5 pixel-center correction',
 'KITTI first raw run externally terminated with code 143 after 13 groups; remaining 7 stored in -002; original manifest RUNNING is stale',
 'initial package download interrupted; reused read-only package cache; Middlebury urllib GT 403 retried successfully with curl']}
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
(out/'input_audit.json').write_text(json.dumps({'all_39_raw_file_hashes_valid':True,'raw_depth_finite_positive':True},indent=2))
report=ROOT/'docs/experiments/planar_residual_depth/P1/P1-audit-20261117-001/report.md';report.parent.mkdir(parents=True,exist_ok=True)
lines=['# P1 실제 데이터 입력 준비 결과','', '**판정: 부분 완료 — 평면 마스크 검토·관측 지원·GT 격자 검증이 남음.**','',
 '세 데이터셋의 공식 RGB·GT 아카이브와 VGGT-Ω 512 checkpoint를 확보했다. GT 깊이 값으로 마스크·대응·보정을 선택하지 않았다. 원본 깊이 추론만 수행했으며 B1–B3 보정과 GT 정확도 평가는 아직 수행하지 않았다.','',
 '| 데이터 | 입력 | 원본 깊이 추론 |','|---|---|---|','| Middlebury | training 15 stereo 쌍 | 30장 완료 |','| KITTI 2015 | training 000000–000019 기준 시각 stereo | 40장 완료 |','| ETH3D | office/pipes/delivery area/courtyard 각 5장 | 20장 완료 |','',
 '39개 공동 추론 묶음·90장. 모든 저장 NPZ의 hash·유한 양의 깊이를 검사했다. 호스트 GPU RTX 5060에서 Torch 2.15.0+cu132로 실행했다. 최대 allocated 메모리는 '+f'{max(peaks)/1024**3:.3f} GiB'+'이며 nvidia-smi 전체 장치 사용량과 구분한다.','',
 '## 입력과 실행 통제','',
 '- 처리 긴 변 512, max_size와 16픽셀 배수. KITTI는 공식 전처리에 맞춰 가로세로비를 2 이하로 중앙 crop했다. K에 crop/resize와 픽셀 중심 변환을 적용했다.',
 '- ETH3D는 제공 COLMAP 카메라를 OpenCV 정수 픽셀 중심으로 변환(cx/cy -0.5)했다. 초기 변환 전 결과는 보존하고 canonical 입력에서 제외했다. [좌표 규약](https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h)',
 '- ETH3D 선택은 카메라 중심의 좌표별 median에 가장 가까운 view부터 시작해 farthest-point로 총5장을 고정했다. 서로 멀리 보는 view라 overlap이 부족할 수 있으며 결과를 보고 view를 바꾸지 않았다.',
 '- SIFT 양방향 ratio 0.75, epipolar 1px. ETH3D·KITTI는 전체 track 단위 hash/seed0 split을 사용하고 같은 view 중복 keypoint track은 거절했다. Middlebury의 첫 준비 결과 split도 별도로 보존했다.',
 '- source hash·K/pose 유효성·좌표 왕복 검사를 통과했다. 이는 ETH3D 원본 GT/undistorted 격자 대응 검증 완료를 뜻하지 않는다.',
 '- 초기 Torch 설치 중 네트워크 중단 이후 로컬 UV 캐시의 읽기 전용 패키지를 P1 전용 환경에서 사용했다. 패키지 경로 목록과 requirements.lock.txt를 코드와 함께 보관한다. P0 환경과 분리했다.','',
 '## RGB 기반 후보 — 사용자 검토 전','', '| ID | 후보 | FIT | 보류 | 지원 판정 |','|---|---|---:|---:|---|']
for c in candidates:lines.append(f"| {c['id']} | {c['label']} | {c['support']['fit']} | {c['support']['heldout']} | {c['support_status']} |")
lines+=['','후보는 RGB만 보고 표시했다. GT·raw 깊이·법선은 후보 선택에 사용하지 않았다. 개수는 view0 지지이며 다른 view의 동일 평면 마스크 검토 전이다. 모두 현재 개수 기준 미달이며 자동 보정하지 않는다. 마스크를 더 넓히거나 threshold를 낮춰 성공 사례로 만들지 않는다. 검토 그림은 `results/planar_residual_depth/review/P1-candidates-v002/candidate_review.png`다.',
 '', '## 남은 작업','', '사용자 검토를 통해 동일 물리 평면 마스크를 다중시점에서 확정하고, 면적·시차·FIT-only scale을 동결해야 한다. GT depth adapter의 왜곡·격자·depth 정의를 검증한 뒤 P1 완료로 승격한다. 현재는 이 조건들이 충족되지 않았으므로 P2 최적화를 실행하지 않았다.',
 '', '## 실행 보존·재개','', *['- '+i for i in summary['preserved_issues']], '', '정확한 canonical 입력·raw run과 샘플별 hash는 `results/planar_residual_depth/P1-audit-20261117-001/summary.json`에 있다. KITTI raw -001의 최초13개와 -002의 나머지7개를 합쳐20개임을 검증했다. 중단 실행을 완료로 바꾸지 않았다.',
 '', '코드: `experiments/planar_residual_depth/P1/`. 원본 depth·confidence·예측 K/pose·FIT/보류 track·RGB·환경·checkpoint hash는 각 run에 분리 저장했다. 실제 GT 개선·평면 방향 개선·장기 SLAM 효과는 아직 평가하지 않았다.']
report.write_text('\n'.join(lines)+'\n')
with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write('\n- P1 입력 준비: **PARTIAL** — [90장 추론·후보 지원·남은 검증](P1/P1-audit-20261117-001/report.md). P2 미실행.\n')
print(json.dumps({'raw_groups':len(records),'frames':frames,'peak_GiB':max(peaks)/1024**3,'report':str(report)},ensure_ascii=False))
