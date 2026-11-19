# P32 평면 패치 검색 및 루프 클로저 파일럿

## 구현과 실행 범위

[구현 계획](../../planar_patch_slam_implementation_plan.md)의 패치 지도, 시야 인덱스, ROI 매칭과 RGB-D 루프 재생 경로를 구현했다. P31 OpenLORIS 앞·뒤 구간과 Bonn 계단 산출물을 읽었다. OpenLORIS 참조 궤적은 공개 아카이브의 `groundtruth.txt`와 카메라 외부 파라미터에서 얻었다. 위치는 선형 보간하고 회전은 Slerp를 적용했다. [OpenLORIS 참조 궤적 설명](https://lifelong-robotic-vision.github.io/dataset/scene#ground-truth)

| 코드 | 역할 |
|---|---|
| [patch_map.py](../../../experiments/planar_residual_depth/P32/patch_map.py) | 패치·관측 ID, 3D 경계 루프와 구멍, 원본 삼각형, ORB 기술자, 병합 근거 저장 |
| [spatial.py](../../../experiments/planar_residual_depth/P32/spatial.py) | 시야 AABB 전수 검사·BVH·BSP 및 다각형 깊이 순서 검사 |
| [benchmark.py](../../../experiments/planar_residual_depth/P32/benchmark.py) | 원본 추정 깊이로 전체 이미지·ROI·장소 후보 매칭 비교 및 포즈 오차 주입 |
| [replay.py](../../../experiments/planar_residual_depth/P32/replay.py) | 센서 깊이로 RGB-D 오도메트리, 장소 후보 검색과 기하 검증, SE(3) 루프 제약 및 포즈 그래프 최적화 |
| [check.py](../../../experiments/planar_residual_depth/P32/check.py) | 시야 질의 일치, 폴리곤 구멍·가림, 포즈 그래프 회귀 검사 |

지도는 서로 다른 프레임의 평면을 원본 다중시점 지지와 양방향 겹침으로 연결했다. 앞 복도 45→39개, 뒤 복도 32→29개, 계단 28→20개였다. 이 수치는 자동으로 연결한 관측 수에 해당한다. 같은 물리 표면의 수동 정답으로 병합 정밀도를 측정하는 작업은 남았다. 지도 패치는 관측별 유한 윤곽을 함께 보존한다. 단일 정제 윤곽으로 재테셀레이션, 공유 변의 기하 검증, 온라인 분할과 지도 갱신·서브맵 간 Sim(3)은 후속 단계다.

## 검색 결과

세 지도 모두 동일 시야에서 선형 검사·BVH·BSP가 같은 AABB 후보를 반환했다. 패치가 20–39개인 지도에서는 벡터화한 선형 검사의 p50이 0.016–0.020 ms로 가장 빨랐다. BSP는 0.037–0.080 ms, BVH는 0.056–0.101 ms였다. 대규모 지도나 갱신 중 지도에서의 속도 우위는 측정해야 한다.

| 실행 | 전체 이미지 p50 | 패치 ROI p50 | ROI 검증 대응 유지율 | ROI 실패 시 전체 이미지 p50 | 재시도 포함 유지율 |
|---|---:|---:|---:|---:|---:|
| OpenLORIS 앞 | 79.74 ms | 21.52 ms | 14/42 | 70.03 ms | 42/42 |
| OpenLORIS 뒤 | 38.81 ms | 8.35 ms | 1/37 | 47.72 ms | 37/37 |
| Bonn 계단 | 97.57 ms | 19.67 ms | 17/24 | 79.86 ms | 24/24 |

유지율의 분모는 원본 깊이와 전체 이미지 특징점으로 기하 검증에 성공한 쌍이다. p50은 특징점을 미리 저장한 뒤 질의마다 후보 조회와 검증을 수행한 시간이다. 전체 이미지 특징점 추출이나 VGGT 추론은 제외했다. 오차 주입은 시야 포즈를 영상 깊이 중앙값의 2·5·10%만큼 이동하고 2·5·10° 회전하는 조건으로 기록했다. 장소 후보 상위 3개와 ROI를 결합한 조건은 뒤 복도에서 기준 대응을 유지하지 못했다.

## 루프 재생 결과와 채택 판단

시간 순서대로 재생한 OpenLORIS 두 구간은 각각 8프레임 모두 추적됐다. 4프레임 초과의 간격과 참조 상대 위치 0.8 m·회전 35° 기준을 만족하는 재방문 후보는 두 구간 모두 0개였다. 역방향 통제 재생에서는 같은 입력 영상을 역순으로 다시 사용했고 각 15프레임을 추적했다. 검색 인덱스는 오프라인에서 만들었으며 재생의 후보 ROI는 그 후보 키프레임 자체의 관측 윤곽을 사용했다.

| 역방향 통제 실행 | 전체 이미지 판정 | 패치만 판정 | 패치 실패 시 전체 이미지 재시도 |
|---|---|---|---|
| OpenLORIS 앞 | 정답 6/6 질의, 오탐 0 | 정답 5/6, 오탐 0 | 정답 6/6, 오탐 0 |
| OpenLORIS 뒤 | 정답 6/6 질의, 오탐 0 | 정답 5/6, 오탐 0 | 정답 5/6, 오탐 1 |

시야만 제한한 매칭은 기준 대응을 크게 잃었고 재시도 경로도 속도 이점이 일관되지 않았다. 뒤 복도 통제 재생의 재시도 경로에서 루프 오탐 1건이 있었다. 이 결과로 실제 로봇 SLAM의 기본 루프 경로는 활성화하지 않는다. 독립 재방문 녹화 및 기존 SLAM 엔진 연결은 남아 있다.

## 재실행

저장 파일은 `map.json`, `geometry.npz`, `mesh.ply`, `observation_evidence.npz`, `features.npz`다. 기존 실행은 지도 `P32-map-<scene>-20261118-003`, 비교 `P32-bench-<scene>-20261118-005`, 재생 `P32-replay-<scene>-<mode>-20261118-004`에 보존했다.

```bash
export OPENBLAS_NUM_THREADS=1
p=experiments/planar_residual_depth/P32
r=results/planar_residual_depth
ref=data/planar_residual_depth/P31/corridor-subset-001
.venvs/p1/bin/python "$p/check.py"
.venvs/p1/bin/python "$p/patch_map.py" "$r/P31d-openloris-early-20261118-001" "$r/P32-map-openloris-early-local-001" --manifest "$ref/early/manifest.json"
.venvs/p1/bin/python "$p/benchmark.py" "$r/P32-map-openloris-early-local-001" "$r/P32-bench-openloris-early-local-001"
.venvs/p1/bin/python "$p/replay.py" "$r/P32-map-openloris-early-local-001" "$r/P32-replay-openloris-early-local-001" "$ref/groundtruth.txt" "$ref/trans_matrix.yaml"
```

역순 통제 실험은 새로운 출력 경로와 `--controlled`를 사용한다.

## OpenLORIS 전체 50초의 2초 간격 재생

원본 1,500장 중 첫 타임스탬프를 기준으로 2초마다 가장 가까운 RGB를 골랐다. 26장의 첫 장부터 마지막 장까지 50.011330초이며 실제 인접 간격은 약 1.9744–2.0084초였다. RTX 5060 호스트에서 26장을 공동 추론했다. P31은 평면 패치 5개를 채택했고 센서 유효 평가 픽셀 1,389,288개에서 깊이 MAE는 경계 필터 기준 39.199 cm, 전 confidence 투영 39.214 cm였다.

- `P32-openloris-2s-strict-20261118-002`는 2초 RGB-D 키프레임만 사용했다. 26장 중 10장을 정합했으며 11번째 키프레임에서 추적이 끊겼다.
- `P32-openloris-2s-bridged-20261118-003`은 실패 구간 4곳에서만 약 0.5초 또는 0.17초 간격의 중간 RGB-D를 포즈 연결에 사용했다. 26개 키프레임을 모두 정합했다. 중간 프레임의 점은 지도 렌더에서 제외했다. ATE RMSE는 4.221 m, 연속 키프레임 RPE RMSE는 1.041 m이며 이번 시퀀스에서 추가된 루프 제약은 0개였다.

두 실행에 `replay.json`과 `registered_pointmap.png`를 보존했다. 그림은 정합된 원본 RGB-D 센서 점군이다.

`P32-openloris-2s-bridged-20261118-003/registered_pointmap.ply`는 그림과 같은 `full` 포즈 및 RGB-D 26장의 유효 샘플 2,221,121개를 2cm 시각화 다운샘플링 전에 저장한 파일이다. `P32-openloris-2s-map-20261118-001/mesh.ply`는 채택한 평면 패치의 메시다. VGGT-Ω 전체 구조에서 평면 투영 효과를 보려면 `P32-openloris-2s-p31-20261118-001`의 `baseline.ply`, `low_only.ply`, `all_confidence.ply`를 같은 시점에서 비교할 수 있다.

## 1초 간격 RGB-D 재실험

같은 원본 약 50초 구간에서 목표 1초마다 가장 가까운 RGB-D 51장을 추출했다. 실제 간격은 0.9720–1.0067초였다. [spacing.py](../../../experiments/planar_residual_depth/P32/spacing.py)는 P32와 동일한 ORB, PnP 및 깊이 검증으로 오도메트리를 계산한다. 새 VGGT-Ω 추론과 루프 최적화는 실행 범위에 포함하지 않았다. `P32-openloris-2s-spacing-control-20261119-001`은 기존 strict 2초 실행의 입력 RGB, 추적 중단 위치, 자세 오차를 재현했다.

- `P32-openloris-1s-strict-20261119-001`은 20/51장을 정합하고 기존과 같은 19.993초 프레임에서 중단됐다. 직전 프레임과의 ORB 대응 23개에서 기하 인라이어 21개가 나왔으나 영상 내 분포 면적 0.507%가 기준 0.8%에 못 미쳤다. 2초 대조군과 공유하는 성공 프레임 10장에서 RPE RMSE는 1.598→0.754m였다.
- `P32-openloris-1s-bridged-20261119-001`은 실패한 3구간에서 약 0.17초 간격 중간 RGB-D로 포즈를 연결해 51/51장을 정합했다. 기존 2초 bridged와 동일한 26개 타임스탬프에서 ATE RMSE는 4.221→2.965m, RPE RMSE는 1.041→0.619m였다. 1초 전체 51장 자체의 ATE는 2.909m다. 1초 간격과 사용한 중간 프레임 경로가 함께 달라졌으므로 전체 구간의 개선은 두 요인의 결합 결과다.

두 1초 실행의 `registered_pointmap.ply`와 `registered_pointmap.png`는 정합된 키프레임만 포함한다. 전체 구간 PLY는 4,357,077점이다.
